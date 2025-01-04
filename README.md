# “WakeWake” 駕駛人睡意偵測警示系統

Author:
邱裕軒 Yuhsuan Chiu (https://github.com/yuxuankuu/)
112453013 (StudentID)
---

# 專案簡介

本專案透過 IoT 與 Raspberry Pi，實現駕駛人睡意偵測警示系統，為了解決疲勞駕駛產生的意外傷故，透過疲勞駕駛偵測、警示控制，以及提供數據可視化功能。

## 主要目標

- 使用樹莓派連接鏡頭，透過影像辨識出一些特定行為（如閉眼睛、打哈欠等）。
- 觀察睡意行為進行視覺及聽覺警示。
- 提供感測資料提供後續分析。

## 應用情境

- 駕駛人常需進行數小時甚至數十小時的長途駕駛，尤其是夜間行車。
- 預防疲勞駕駛事故，降低人員傷亡。
- 記錄疲勞駕駛數據，提供可量化的資料測量記錄。

**1至10月國道死亡自撞占37% 多因恍神、疲勞駕駛**

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image.png)

交通部高速公路局統計，113年1月至10月國道已發生57起死亡交通事故，其中肇事型態屬於「自撞」為21件（占37%），肇因最多為恍神、分心、疲勞駕駛等。（高公局提供）

## 系統功能

### 視覺偵測

用視覺辨識偵測眼睛和嘴唇閉合狀態

### 警報提示

觸發警示燈和蜂鳴器聲音警報

### 分析報表

在網頁上顯示駕駛狀態以日期時間為記錄

## 系統架構

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_1.png)

### **硬體需求**

- **樹莓派型號**：Raspberry Pi 4 或以上版本
- **外部裝置**：
    - Pi Camera 鏡頭模組
    - 七段顯示器 (Optional)
    - LED 元件和電阻
    - 蜂鳴器
    - SD Card (至少32GB，用於日誌記錄和備份)
    - 電源供應器
    - 麵包板

### **軟體技術**

- **語言框架**：
    - Python 3.7 或以上版本
    - Flask（後端 API 開發）
- **函式庫**：
    - `dlib`：機器學習、人臉偵測
    - `gpiozero`：GPIO 控制
    - `cv2`：OpenCV影像處理
    - `numpy`：數據處理
    - `matplotlib`：數據可視化

### Diagram

(使用 [fritzing](https://fritzing.org/) 繪製)

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_2.png)

### 系統示意圖

將樹莓派安裝在車內，使鏡頭面對駕駛人位置安裝在方向盤後方。

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/demo_1.jpg)

偵測系統啟動時(綠色LED閃爍)並出現歡迎語"HELLO"，偵測系統運行中(綠色LED恆亮)。

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/demo_2.jpg)


# 實作範例

## 1. 環境安裝

### 作業系統安裝

透過 **Raspberry Pi Imager** (按此[連結](https://www.raspberrypi.com/software/)下載並安裝) 將 Raspberry Pi 作業系統安裝到 microSD 卡，此範例使用預先準備的 Image **`2021-05-07-raspios-buster-armhf.img`。**

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_3.png)

### 系統和套件更新

透過 RealVNC 登入 Raspberry Pi 系統後，進行軟體更新以及相關操作。

```bash
sudo apt update && sudo apt upgrade -y
```

安裝基本開發工具

- 樹莓派通常自帶 Python，但需要確認並安裝所需版本，使用以下命令檢查套件版本：
    
    ```bash
    python3 --version
    pip3 list
    ```
    
- 樹莓派上運行 dlib 和 OpenCV 需要 CMake 和其他開發工具 (參考 Ref #1)：
    
    ```bash
    cd ~/
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4.tar.gz
    tar xvzf cmake-3.14.4.tar.gz
    cd ~/cmake-3.14.4
    ./bootstrap
    ```
    
    成功畫面：
    
    ![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/result_1.png)
    
    執行編譯：
    
    ```bash
    make -j4
    ```
    
    成功畫面：
    
    ![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/result_2.png)
    
    執行安裝：
    
    ```bash
    sudo make install
    ```
    
    成功畫面：
    
    ![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/result_3.png)
    
- 安裝 dlib 和 OpenCV
    
    ```bash
    pip3 install dlib
    pip3 install opencv-python
    ```
    
- 安裝其他 Dependencies
    
    ```bash
    pip3 install numpy scipy imutils flask
    ```
    

### 準備模型

到 Dlib Library [http://dlib.net/files/](http://dlib.net/files/) 下載人臉特徵點模型 **shape_predictor_68_face_landmarks.dat**  (參考 Ref #2)

`shape_predictor_68_face_landmarks.dat`是由 Dlib Library 提供的預訓練模型，專門用於人臉特徵點檢測。它可以識別人臉上的 68 個特徵點，例如眼睛、眉毛、鼻子、嘴巴和下巴的關鍵位置。這些特徵點常用於人臉識別、表情分析、臉部特徵對齊等應用。

![**The 68 landmarks detected by dlib library. This image was created by Brandon Amos of CMU who works on OpenFace.**](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_4.png)

**The 68 landmarks detected by dlib library. This image was created by Brandon Amos of CMU who works on OpenFace.**

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

確保此文件與程式碼放在相同目錄下。

<aside>
💡

臉部模型測試把眼鏡拿下來識別精確度大幅提升。

</aside>

**眼睛距離比 EAR（Eye Aspect Ratio）計算** (參考 Ref #3)

判斷眼睛是否閉合的計算公式，可以用眨眼的檢測。

```python
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]) #眼睛垂直距離
    B = dist.euclidean(eye[2], eye[4]) #眼睛垂直距離
    C = dist.euclidean(eye[0], eye[3]) #眼睛水平距離

    ear = (A + B) / (2.0 * C)

    return ear
```

### 安裝鏡頭

 (參考 Ref #4)

1. 確保 Raspberry Pi 已關機。 
2. 找到鏡頭模組接口，輕輕拉起端口塑膠夾的邊緣，插入鏡頭模組帶狀電纜。
3. 確保帶狀電纜底部的連接器面向連接埠中的接點，將塑膠夾推回原位。
4. 重新啟動 Raspberry Pi。

執行以下指令測試 Camera 功能：

```bash
raspistill -o ~/Desktop/test.jpg
```

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_5.gif)

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_6.png)

### 安裝其他元件

增加七段顯示器線路 (使用 [tinkercad](http://www.tinkercad.com) 繪製)

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_7_1.png)

![image.jpg](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_7_2.jpg)

## 2. 執行程式

執行偵測程式：

```bash
python3 drowsiness_yawn.py
```

![image.jpg](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_8.jpg)

執行後台程式 app.py，用 Flask 實作讀取疲勞和和打哈欠警告寫入日誌檔案`alert_log.log`，並製作成統計報告，以方便後續分析應用，可以做為駕駛人調度參考。

```bash
python3 app.py
```

![image.png](https://github.com/yuxuankuu/IoT-RPi-Project/blob/main/image/image_9.png)

# 改進建議

- 目前的介面較為簡單，使用 CSS 或 Bootstrap 來提升頁面美觀。
- 使用 Bar Chart 圖表或儀板表來顯示警報次數的分佈。
- 增加額外功能：酒精測試
對駕駛者的酒精測試模組，並將檢測結果與疲勞檢測結合，提供更全面的監控。

---

# 參考 References

1. **梁藝鐘 Openvino with NCS2 ON Raspi-4** [https://hackmd.io/HV6hQ2PHSiWlrRsfxC10SA](https://hackmd.io/HV6hQ2PHSiWlrRsfxC10SA)
2. **基于opencv和shape_predictor_68_face_landmarks.dat的人脸识别监测** [https://blog.csdn.net/monster663/article/details/118341515](https://blog.csdn.net/monster663/article/details/118341515)
3. **python dlib学习（十一）：眨眼检测** [https://blog.csdn.net/hongbin_xu/article/details/79033116](https://blog.csdn.net/hongbin_xu/article/details/79033116)
4. **Connect the Camera Module** [https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/2](https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/2)