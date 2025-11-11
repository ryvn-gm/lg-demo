import os
import json
import pymysql
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch

# 載入 .env 檔案中的環境變數
# override=True 表示如果變數已存在，會用 .env 裡的覆蓋掉
load_dotenv(override=True)

# --- 1. 內建工具：Tavily 網路搜尋 ---
# max_results=5 表示最多回傳 5 筆搜尋結果
search_tool = TavilySearch(max_results=5, topic="general")


# --- 2. 資料庫工具 (SQL Interactor) ---
# TODO: 實作資料庫工具


# --- 3. Python 程式碼執行工具、 csv 載入工具 ---

# ⚠️⚠️⚠️ --- 安全性警告 --- ⚠️⚠️⚠️
# 以下兩個工具 (`python_inter`, `fig_inter`) 使用了 `exec` 和 `eval`，
# 這表示它們可以執行任意的 Python 程式碼。
# 如果 LLM 被誘導產生惡意程式碼 (例如刪除檔案、竊取密碼)，
# 這些程式碼將會被「直接執行」，造成嚴重安全風險。
#
# ❗️ 在生產環境中，**絕對不要**在沒有沙盒 (Sandbox) 的情況下執行此類程式碼。
# ❗️ 請務必將執行環境隔離 (例如使用 Docker 容器或 python-sandbox 套件)。
# ⚠️⚠️⚠️ --- 安全性警告結束 --- ⚠️⚠️⚠️


class PythonCodeInput(BaseModel):
    py_code: str = Field(
        description="一段合法的 Python 程式碼字串。例如 '2 + 2' 或 'x = 3\ny = x * 2' 或 'print(df.head())'。"
    )


@tool(args_schema=PythonCodeInput)
def python_inter(py_code: str):
    """
    [工具] 執行「非繪圖類」的 Python 程式碼。

    當使用者需要執行 Python 腳本、進行資料處理、統計計算或查看變數內容時，
    請呼叫此函式。

    注意：本函式「無法」處理繪圖。如果需要繪圖，請呼叫 `fig_inter`。
    """
    print(f"🚀 [工具] 正在執行 python_inter: {py_code}")

    # 取得當前的全域變數範圍
    g = globals()

    try:
        # 1. 嘗試當作「表達式 (expression)」執行 (例如: '2 + 2', 'df.shape')
        # `eval` 會回傳表達式的計算結果
        return str(eval(py_code, g))

    except Exception as e_eval:
        # 2. 如果 `eval` 失敗 (通常因為它是「陳述式 (statement)」，例如 'x = 5' 或 'import os')

        # 記錄執行前的全域變數
        global_vars_before = set(g.keys())

        try:
            # 嘗試當作「陳述式」執行
            exec(py_code, g)

            # 記錄執行後的
            global_vars_after = set(g.keys())

            # 找出新建立的變數
            new_vars = global_vars_after - global_vars_before

            if new_vars:
                # 如果有新變數，回傳新變數和它們的值
                result = {var: g[var] for var in new_vars}
                print("✅ 程式碼順利執行，並建立新變數...")
                return f"程式碼執行成功，建立或更新了變數: {', '.join(new_vars)}. (值: {str(result)})"
            else:
                # 如果沒有新變數 (例如 'print()' 或修改了現有變數)
                print("✅ 程式碼順利執行...")
                return "✅ 程式碼已順利執行。"

        except Exception as e_exec:
            # 如果 `exec` 也失敗了，回報雙重錯誤
            return f"❌ 程式碼執行失敗。\n[Eval 錯誤]: {e_eval}\n[Exec 錯誤]: {e_exec}"


class FigCodeInput(BaseModel):
    py_code: str = Field(
        description="要執行的 Python 繪圖程式碼 (必須使用 matplotlib/seaborn)。"
    )
    fname: str = Field(
        description="要儲存的圖像「物件變數名稱」(例如 'fig')，程式碼中必須將圖像賦值給此變數。"
    )


@tool(args_schema=FigCodeInput)
def fig_inter(py_code: str, fname: str) -> str:
    """
    [工具] 執行 Python 程式碼以「產生並儲存圖片」。

    當使用者需要使用 Python 進行視覺化繪圖任務時，請呼叫此函式。

    ⚠️ 重要使用規則：
    1. 所有繪圖程式碼**必須**建立一個圖像物件 (Figure) 並將其賦值給 `fname` 參數指定的變數名。
       (例如，如果 `fname="my_plot"`，程式碼中要有 `my_plot = plt.figure()`)
    2. 必須使用 `fig = plt.figure()` 或 `fig, ax = plt.subplots()` 來建立圖像。
    3. **絕對不要**使用 `plt.show()`，否則會導致後台崩潰。
    4. 程式碼末尾建議呼叫 `fig.tight_layout()` (或 `fname` 對應的變數) 確保佈局緊湊。
    5. 圖表中的所有文本 (標題、標籤) 建議使用英文，以避免字體問題。

    範例程式碼 (假設 fname='fig'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_title("My Plot")
    fig.tight_layout()
    """
    print(f"🚀 [工具] 正在執行 fig_inter (儲存為 {fname})...")

    # 儲存當前的 matplotlib 後端，並切換到 'Agg' (非互動式後端)
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    # 準備一個局部的執行環境，預先載入必要的函式庫
    local_vars = {"plt": plt, "pd": pd, "sns": sns}
    g = globals()  # 也傳入全域變數 (例如先前 `extract_data` 載入的 df)

    base_dir = Path(__file__).parent
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)  # 自動建立 images 資料夾

    try:
        # 執行繪圖程式碼
        exec(py_code, g, local_vars)

        # 從局部變數中取得繪圖物件
        fig = local_vars.get(fname, None)

        if fig and (isinstance(fig, plt.Figure) or isinstance(fig, sns.JointGrid)):
            image_filename = f"{fname.replace(' ', '_')}.png"  # 檔名清理
            abs_path = images_dir / image_filename  # 絕對路徑

            rel_path = abs_path.relative_to(base_dir)

            # 儲存圖片
            fig.savefig(abs_path, bbox_inches="tight")

            # 回傳「相對路徑」(給前端或 Markdown 使用)
            return f"✅ 圖片已成功儲存，路徑為: {rel_path.as_posix()}"
        else:
            return f"⚠️ 圖像物件 '{fname}' 未在程式碼中找到，或該物件不是有效的 matplotlib 圖形物件。"

    except Exception as e:
        return f"❌ 執行繪圖程式碼時發生錯誤：{e}"

    finally:
        # 恢復原本的 matplotlib 後端
        plt.close("all")  # 關閉所有圖形，釋放記憶體
        matplotlib.use(current_backend)


class LoadCSVSchema(BaseModel):
    file_path: str = Field(
        description="本地 CSV 檔案的路徑 (例如 'data/my_data.csv')。"
    )
    df_name: str = Field(
        description="指定一個變數名稱 (字串)，用於儲存讀取到的 pandas DataFrame。"
    )


@tool(args_schema=LoadCSVSchema)
def load_csv_data(file_path: str, df_name: str) -> str:
    """
    [工具] 讀取本地 CSV 檔案，並將其「存成 pandas DataFrame 變數」。

    當使用者需要開始分析一個 CSV 檔案時，這是必須呼叫的「第一步」。

    ⚠️ 注意：本函式會使用 `globals()[df_name]` 將 DataFrame
    存入全域變數，這是一種「有狀態」的設計。

    :param file_path: 本地 CSV 檔案的相對或絕對路徑。
    :param df_name: 將提取的表格儲存為本地 pandas DataFrame 時的變數名稱。
    :return: 一個字串，說明 CSV 讀取和儲存的結果。
    """
    print(f"🚀 [工具] 正在執行 load_csv_data: 將 {file_path} 存為 {df_name}")

    try:
        # 檢查檔案是否存在
        if not os.path.exists(file_path):
            return f"❌ 錯誤：找不到檔案 {file_path}"

        # 使用 pandas 讀取 CSV
        df = pd.read_csv(file_path)

        # ⚠️ 存入全域變數
        globals()[df_name] = df

        print(f"✅ CSV 成功載入並儲存為全域變數：{df_name}")
        return f"✅ 成功從 {file_path} 載入資料，並建立 pandas DataFrame 變數 `{df_name}`。 (共 {len(df)} 筆資料)"

    except Exception as e:
        return f"❌ 執行 load_csv_data 時發生錯誤：{e}"


# --- 4. 系統提示 (System Prompt) ---

prompt = """
你是一名專業、資深的「智慧數據分析師」(AI Data Analyst)。
你的任務是協助使用者高效、準確地完成數據相關工作。
使用者的資料來源「只有」本地的 CSV 檔案。

你精通以下任務，並會依照指示使用對應的工具：

1.  **CSV 資料載入 (`load_csv_data`)**:
    * **時機**: 這是所有分析的「第一步」。當使用者說「開始分析」或提到 CSV 檔案時，你必須先呼叫這個工具。
    * **動作**: 你會呼叫 `load_csv_data`，並指定 CSV 檔案的路徑 (例如 `'my_data.csv'`) 和一個「變數名稱」(例如 `df`)。
    * **結果**: 這會在 Python 環境中建立一個名為 `df` 的 pandas DataFrame 變數。

2.  **Python 程式碼執行 (`python_inter`)**:
    * **時機**: 當使用者需要執行「非繪圖類」的 Python 程式碼時。
        * 例如：資料處理 (使用 pandas)、統計計算、查看 DataFrame (如 `print(df.head())`)、定義變數等。
    * **動作**: 呼叫 `python_inter`。

3.  **Python 視覺化繪圖 (`fig_inter`)**:
    * **時機**: 當使用者需要「繪製圖表」時 (例如：長條圖、散點圖、熱力圖等)。
    * **動作**: 你會生成完整的 Python 繪圖程式碼 (使用 matplotlib 或 seaborn)，並指定一個「圖像物件變數名稱」(例如 `fig`)。
    * **重要規則**:
        * 你「必須」在程式碼中建立圖像物件 (例如 `fig, ax = plt.subplots()`)。
        * 你「絕對不能」呼叫 `plt.show()`。
        * 你「必須」在程式碼結尾加入 `fig.tight_layout()`。

4.  **網路搜尋 (`search_tool`)**:
    * **時機**: 當使用者詢問與數據分析「無關」的問題時 (例如：即時新聞、天氣、一般知識)。

---

**工作流程與準則**:

* **第一步**: 永遠先呼叫 `load_csv_data` 將 CSV 載入為 DataFrame (例如 `df`)。如果使用者沒有提供路徑，你應該主動詢問。
* **數據依賴**: 執行 `python_inter` 或 `fig_inter` 之前，請確保所需的 `df` 已經透過 `load_csv_data` 載入。
* **回答語言**: 所有回答都必須使用**「繁體中文」**，保持清晰、禮貌、專業。
* **展示結果**:
    * 如果成功生成了圖片 (來自 `fig_inter`)，你**必須**在回答中使用 Markdown 格式插入圖片。
        * 範例： `![用戶流失的類別特徵分析](這邊是圖片的相對路徑，並且前面要以 http://localhost:2024/ 開頭，因為是後端伺服器的路徑)`
    * **絕對不要**只輸出圖片的路徑文字。
* **風格**: 保持專業、簡潔、以數據為導向。
"""

# --- 5. 建立 Agent ---

# 整理所有工具
tools = [search_tool, python_inter, fig_inter, load_csv_data]

# 建立模型 (假設使用 gpt-4o-mini)
model = ChatOpenAI(model="gpt-4o")

# 建立 Agent
agent = create_agent(model=model, tools=tools, system_prompt=prompt)
