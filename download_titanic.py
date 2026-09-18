import kagglehub
import shutil
from pathlib import Path

# Kaggle에서 Titanic 데이터셋 다운로드
download_path = Path(
    kagglehub.dataset_download("heptapod/titanic")
)

# 현재 프로젝트 폴더 기준 저장 위치
project_dir = Path.cwd()
target_dir = project_dir / "data" / "titanic"

target_dir.mkdir(parents=True, exist_ok=True)

# 다운로드된 파일들을 프로젝트 폴더로 복사
for file in download_path.iterdir():
    if file.is_file():
        shutil.copy2(file, target_dir / file.name)

print("원본 다운로드 위치:", download_path)
print("프로젝트 저장 위치:", target_dir)

print("\n파일 목록:")
for file in target_dir.iterdir():
    print("-", file.name)
