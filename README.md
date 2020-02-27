# TableRecognition
Recognize tables from images and restore them into word.

### How to run
1. python server.py  
   Load the unet model to extract table lines from the input image 
2. python test.py  
   Feed the input image
3. python image2word.py  
   Restore table use opencv & python-docx

### Tips  
Step 1 & 2 are not necessary if you have quite neat PDF images, meanwhile this project
can't deal with some complex samples like tortuous and colorful receipts, I am still working on it.
  
### To do
I am handling table recognition like this [image](https://pic3.zhimg.com/a1b8009516c105556d2a2df319c72d72_b.jpg), struggling with the dataset.
Optimistically, there could be a radical change in weeks. If you are researching page 
layout and table recognition, please contact me.[lizongxi1995@gmail.com](lizongxi1995@gmail.com)

### Reference and some useful projects
1. [https://github.com/chineseocr/table-ocr.git](https://github.com/chineseocr/table-ocr.git)
2. [https://github.com/weidafeng/TableCell.git](https://github.com/weidafeng/TableCell.git)
3. [腾讯表格识别方案简述](https://blog.csdn.net/Tencent_TEG/article/details/94080906?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)
4. [OpenCV-检测并提取表格](https://blog.csdn.net/yomo127/article/details/52045146?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)