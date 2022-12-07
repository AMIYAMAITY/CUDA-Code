***Batch DCT RUN***

**Step1:** opencv C++ and GPU driver should install.

**Step2:** Make sure export opencv lib path. ` export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/opencv/lib/ `

**Step3:** Compile kernel code. `nvcc kernel.cu -o kernel -I /opt/opencv/include/opencv4/  -L /opt/opencv/lib/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio `

**Strep4:** Once it's compiled kernel executable file will generate and then execute it. `./kernel`

**Step5:** After execute the kernel file DCT hex will show- 

DCT hex: fa813f3421af1a6a

DCT hex: c9bac4b197b39096

DCT hex: a7a45a929a99d9cc

DCT hex: c9bac4b197b39096

DCT hex: a7a45a929a99d9cc