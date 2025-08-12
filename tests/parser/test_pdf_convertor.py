import os
import sys
# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
#from services.dataparse.pdf_convertor import convert_to_pdf

if __name__ == "__main__":
    # 测试Office文件转换为PDF的代码
    input_file = r"/home/chris/docs/resume.docx"
    output_file = r"/home/chris/docs/resume.pdf"
    #convert_to_pdf(input_file, output_file)