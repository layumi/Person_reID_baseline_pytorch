The whole process of using colab to train Person_reID
=====
## 1.Create a colab notebook on Google Drive
If this is your first time associating Google Drive with colab :  
Click New -> More -> Connect more apps, find Google Colaboratory, and connect.  

Create a new colab notebook :  
Click New -> More ->Google Colaboratory  
## 2. Use free GPU  
Edit-> Notebook settings, select GPU   
## 3. Execute the command (press alt+enter to execute quickly)  
colab is equivalent to jupyter notebook, you can run python code directly, This notebook can also execute some commands under linux, because this is actually a linux virtual machine, but when you execute linux commands, you must add ! in front of it, such as:
``` 
!ls, !pwd.
```
## 4. Due to the needs of the project, pytorch needs to be installed here and related configuration is performed. Enter the following code in colab:  
```
!pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
!python -m pip install matplotlib
!pip install  pretrainedmodels
!pip install timm
```
## 5. Hang colab on Google Drive, so that you can save the results to Google Drive:
```
from google.colab import drive
drive.mount('/content/drive/')
```
Run this code, it will display a connection verification, click to verify.  
## 6. Enter the path
```
path = "/content/drive/MyDrive"
import os
from google.colab import drive
os.chdir(path)
os.listdir(path)
```
## 7. Download the code directly from GitHub and Prepare the data set:  
```
!git clone https://github.com/layumi/Person_reID_baseline_pytorch.git
``` 
Download Market-1501 to your computer, then upload the compressed package to Google Driver  
Then you can see the market data set under the driver file of colab, enter the code, unzip the data set to the project 
```
!unzip '/content/drive/MyDrive/Market-1501-v15.09.15.zip' -d '/content/drive/MyDrive/Person_reID_baseline_pytorch'
```
The front is the directory where the compressed package is located, and the back is the directory to save after decompression  
## 8. Enter the operating directory and run the program.  
Need to change first, the corresponding path in prapare.py ,train.py ,test.py and demo.py  
```
# You only need to change this line to your dataset download path_ prepare.py
download_path = '/content/drive/MyDrive/Person_reID_baseline_pytorch/Market-1501-v15.09.15'
```
```
# You only need to change this line to your Market-1501-v15.09.15/pytorch download path_ train.py
parser.add_argument('--data_dir',default='/content/drive/MyDrive/Person_reID_baseline_pytorch/Market-1501-v15.09.15/pytorch',type=str, help='training dir path')
```
```
# You only need to change this line to your Market-1501-v15.09.15/pytorch download path_ test.py
parser.add_argument('--test_dir',default='/content/drive/MyDrive/Person_reID_baseline_pytorch/Market-1501-v15.09.15/pytorch',type=str, help='./test_data')
```
```
# You only need to change this line to your Market-1501-v15.09.15/pytorch download path_ demo.py
parser.add_argument('--test_dir',default='/content/drive/MyDrive/Person_reID_baseline_pytorch/Market-1501-v15.09.15/pytorch',type=str, help='./test_data')
```
Enter the catalog:
```
%cd /content/drive/MyDrive/Person_reID_baseline_pytorch
```
Data preparation:
```
!python prepare.py
```
Training:

```
! python train.py
```
Test:
```
! Python test.py
```
Evaluation: 
```
!python evaluate_gpu.py
```
Visualization: 
```
!python demo.py --query_index 750 (0-3367)
```
## 9. If you want to download the file to the local, run the following code  
(compress first, then download to the local)
```
import os, tarfile
import os
from google.colab import files
def make_targz_one_by_one(output_filename, source_dir):
  tar = tarfile.open(output_filename,"w")
  for root,dir_name,files_list in os.walk(source_dir): 
    for file in files_list:
      pathfile = os.path.join(root, file)
      tar.add(pathfile)
  tar.close()
  files.download(output_filename)
make_targz_one_by_one('peo', '/content/drive/MyDrive/Person_reID_baseline_pytorch')
```
peo is the name of the compressed file, which can be anything you want. /content/drive/MyDrive/people/person/person reid is the name of the file to be downloaded.

## Note: Every time you reopen colab, you must reinstall the environment and match Google Drive.
```
!pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
!python -m pip install matplotlib
!pip install  pretrainedmodels
!pip install timm
from google.colab import drive
drive.mount('/content/drive/')
```

Colab may not support `torch.compile()`, which accelerates the speed,  and you only needs to disable it in the train and test file. 
https://github.com/layumi/Person_reID_baseline_pytorch/issues/398
