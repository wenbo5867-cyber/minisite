from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import uuid
import shutil
import zipfile
import os

from hand_motion_analysis import run_analysis

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)



@app.route("/")
@app.route("/index")
def index():
    # 获取 open 参数
    open_item = request.args.get("open", "")  # 例如 ?open=Image_analysis

    # 如果需要调用 Python 逻辑
    if open_item == "Image_analysis":
        # 这里可以调用你的 Python 函数处理数据
        print("调用 Image_analysis 相关 Python 逻辑")

    # 渲染模板并传给前端
    return render_template("index.html", open_item=open_item)
@app.route("/run", methods=["POST"])
def run():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 每次任务一个独立 ID（防覆盖）
    task_id = str(uuid.uuid4())[:8]

    upload_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    out_dir = RESULT_DIR / task_id

    out_dir.mkdir(parents=True, exist_ok=True)
    file.save(upload_path)

    # 参数
    lag = int(request.form.get("lag", 2))
    n_joints = int(request.form.get("n_joints", 8))

    try:
        result = run_analysis(
            file_path=upload_path,
            n_joints=n_joints,
            lag=lag,
            out_dir=out_dir
        )
    except Exception as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "status": "ok",
        "task_id": task_id,
        "zip_url": f"/download/{task_id}",
        "excel_url": f"/download_excel/{task_id}"
    })

@app.route("/download_excel/<task_id>")
def download_excel(task_id):
    excel_path = RESULT_DIR / task_id / "results.xlsx"
    if not excel_path.exists():
        return "Excel not found", 404

    return send_file(
        excel_path,
        as_attachment=True,
        download_name="results.xlsx"
    )


@app.route("/download/<task_id>")
def download(task_id):
    zip_path = RESULT_DIR / task_id / "results.zip"
    if not zip_path.exists():
        return "File not found", 404
    return send_file(zip_path, as_attachment=True)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
