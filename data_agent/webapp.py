from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()


images_dir = Path(__file__).parent / "images"
images_dir.mkdir(parents=True, exist_ok=True)  # 確保資料夾存在

app.mount(
    "/images",  # 網頁上的路徑
    StaticFiles(directory=images_dir),  # 實體資料夾的路徑
    name="images",  # 內部名稱
)
