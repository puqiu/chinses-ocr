pip install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple/ ##选择国内源，速度更快
pip install keras==2.0.8 tensorflow==1. tensorflow-gpu==1.0.0rc1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install Cython opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
cd ./ctpn/lib
python setup.py build

