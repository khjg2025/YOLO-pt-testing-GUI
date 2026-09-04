# 初始化Git仓库并推送到GitHub

cd E:\code\yolo26gui
set PATH=C:\Program Files\Git\bin;%PATH%
git init
git config user.name "khjg2025"
git config user.email "khjg2025@example.com"
git remote add origin https://github.com/khjg2025/YOLO-pt-testing-GUI.git
git add .
git commit -m "Initial commit: YOLO Detection and Training GUI"
git branch -M main
git push -u origin main
