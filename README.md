
---

# 基於遷移學習之 COVID-19 放射影像輔助診斷系統開發

## 簡介
本次報告使用深度學習（Convolutional Neural Network, CNN）與遷移學習（Transfer Learning），
對 COVID-19 影像資料進行二元分類（Positive / Negative），並透過混淆矩陣與分類報告評估模型效能。

---

## 環境與套件初始化

**參考資料：https://blog.csdn.net/woshicver/article/details/120398109**

**資料集來源：https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset**

---

## 步驟一：導入開發環境與相關工具庫 (Environment Setup & Libraries Import)

在此步驟中，我們載入了數據處理、視覺化以及建構深度學習模型所需的所有核心套件。

### 1. 數據處理與視覺化 (Data Processing & Visualization)

* **NumPy / Pandas**: 用於高效的數值計算與資料表操作。
* **Matplotlib / Seaborn**: 用於繪製訓練曲線、統計圖表及顯示影像。
* **tqdm**: 提供進度條功能，方便觀察資料處理進度。

### 2. 深度學習框架 (TensorFlow & Keras)

* **Sequential / Layers**: 用於建構神經網路的堆疊結構與各種類型的層（如卷積層、池化層、全連接層）。
* **Optimizers**: 導入了 `Adam`、`SGD`、`RMSprop` 等優化算法。
* **Pre-trained Models**: 載入了經典的遷移學習模型架構，包括 `DenseNet121`、`VGG19` 與 `ResNet50`。

### 3. 影像處理 (Image Processing)

* **PIL / ImageDataGenerator**: 用於讀取、轉換影像，以及實作資料增強 (Data Augmentation)，增加模型的泛化能力。

### 4. 其他配置 (Miscellaneous)

* **Warnings**: 忽略非必要的警告訊息，保持輸出介面整潔。
* **Shuffle**: 用於在訓練前打亂數據順序，避免模型學習到規律性的雜訊。

```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import numpy as np
import pandas as pd
sns.set()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications import DenseNet121, VGG19, ResNet50

import PIL.Image
import matplotlib.image as mpimg
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.preprocessing import image

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from sklearn.utils import shuffle
````
<img width="865" height="628" alt="1" src="https://github.com/user-attachments/assets/8493173a-44f2-42da-8349-4ee26e1e2574" />

---

## 步驟二：連結 Google 雲端硬碟 (Mount Google Drive)

在 Colab 環境中，為了能夠存取儲存在雲端硬碟中的資料集（如影像、CSV 檔案）或儲存訓練好的模型權重，我們需要進行掛載操作。

### 執行說明

* **權限驗證**：執行此儲存格後，系統會跳出視窗要求授權存取 Google Drive。
* **路徑映射**：掛載成功後，雲端硬碟內容將會出現在 `/content/drive/MyDrive` 路徑下。
* **持久化儲存**：相較於 Colab 暫時性的磁碟空間，儲存在 Drive 中的資料在工作階段結束後不會消失。


```python
from google.colab import drive
drive.mount('/content/drive')
```
<img width="422" height="120" alt="2" src="https://github.com/user-attachments/assets/3b44674e-0102-4084-9597-35db0764719b" />

---

## 步驟三：資料解壓與 Metadata 載入 (Data Extraction & Metadata Loading)

為了提高讀取速度，我們將儲存在雲端硬碟中的壓縮檔（`.zip`）解壓縮至 Colab 本地的虛擬磁碟空間（`/content/`），並使用 Pandas 讀取標籤資訊。

### 1. 解壓縮檔案

* **效能最佳化**：直接在雲端硬碟讀取數千張小圖檔速度極慢，解壓至本地空間可大幅提升後續訓練時的 I/O 效率。
* **靜默模式**：使用 `!unzip -q` 指令進行背景解壓，避免過多的輸出訊息佔據介面。

### 2. 資料結構確認

* **路徑檢查**：使用 `os.path.exists` 自動確認 CSV 標籤檔是否存在，確保後續資料處理流程不中斷。
* **Metadata 載入**：讀取 `Chest_xray_Corona_Metadata.csv`，這通常包含了影像檔案名稱、類別（如 Normal, Virus, Bacteria）以及資料分集（Train/Test）等重要資訊。

### 3. 數據概覽

* **Shape 確認**：透過 `train_df.shape` 快速確認資料集的樣本總數與欄位數量。


---

```python
import pandas as pd
from google.colab import drive
import os

#  掛載 Google Drive
drive.mount('/content/drive')

#  解壓縮檔案到 Colab 的暫存空間 (/content/chest_xray)
# 使用 -q 避免解壓時顯示幾千張照片的檔名，使用 -d 指定解壓資料夾
!unzip -q "/content/drive/MyDrive/1226/archive.zip" -d "/content/chest_xray"

# 讀取 CSV 檔案
# 通常 Kaggle 下載的 zip 解壓後，CSV 會在解壓資料夾的根目錄
csv_path = '/content/chest_xray/Chest_xray_Corona_Metadata.csv'

if os.path.exists(csv_path):
    train_df = pd.read_csv(csv_path)
    print("讀取成功！")
    print(f"Dataset Shape: {train_df.shape}")
else:
    print("找不到 CSV 檔案，請檢查解壓後的檔案結構。")
    # 如果找不到，可以用這行指令看看解壓出來到底有哪些檔案
    !ls -R /content/chest_xray
```
<img width="1267" height="681" alt="3" src="https://github.com/user-attachments/assets/d5c8f8d9-9701-4026-9728-2831b07746ff" />

---

## 步驟四：檢視資料表前五筆數據 (Preview Dataset Head)

透過展示資料集的前五列（Top 5 Rows），我們可以確認 Metadata 的欄位定義，並檢察資料是否正確載入。

### 欄位解析說明

通常在 **Chest X-ray (COVID-19/Pneumonia)** 資料集中，您會看到以下關鍵欄位：

* **`X_ray_image_name`**: 對應影像資料夾中的檔案名稱。
* **`Label`**: 第一層分類（如：Normal 或 Pnemonia）。
* **`Dataset_type`**: 標註該樣本屬於 `TRAIN` 還是 `TEST` 集。
* **`Label_2_Virus_category`**: 第二層細分（如：COVID-19, SARS, Streptococcus 等具體病原體）。

### 執行意義

* **驗證資料格式**：確認遺漏值（NaN）的狀況。
* **路徑對接準備**：確認檔名欄位，以便後續將 CSV 中的字串與資料夾中的圖片路徑進行連結。

```python
train_df.head(5)
```
<img width="1160" height="325" alt="4" src="https://github.com/user-attachments/assets/cb2df4e1-113b-4835-9525-0b8bb958991e" />

---

## 步驟五：資料集結構與型態分析 (Dataset Information & Diagnostics)

透過執行 `train_df.info()`，我們對資料框（DataFrame）進行整體的技術性檢查，這對於後續的資料清洗（Data Cleaning）至關重要。

### 檢查重點：

1. **資料筆數 (Total Entries)**：確認樣本總數是否與預期相符。
2. **欄位型態 (Dtype)**：確認類別型資料是否為 `object`（字串），數值型資料是否為 `int` 或 `float`。
3. **缺失值檢查 (Non-Null Count)**：
* 檢查是否有欄位存在大量缺失值（NaN）。
* 在胸部 X 光資料集中，較細的分類（如 `Label_2_Virus_category`）通常會有許多缺失值，因為並非所有病人都有明確的病原體檢測結果。


4. **記憶體使用量 (Memory Usage)**：評估目前的資料規模對系統記憶體的負擔。

```python
train_df.info()
```
<img width="555" height="353" alt="5" src="https://github.com/user-attachments/assets/31e426f0-0d45-44dc-9f66-4d0ce3cfe854" />

---

## 步驟六：缺失值分析與視覺化 (Missing Values Analysis & Visualization)

透過計算各個欄位的缺失值（Null values）數量並將其視覺化，我們可以更直觀地掌握資料集中哪些特徵存在資訊缺漏，這對於決定特徵工程的策略至關重要。

### 檢查重點：

1. **缺失值統計 (Calculating Missing Values)**：利用 `isnull().sum()` 統計每一列中缺失數據（NaN）的總數，確認資料的完整性。
2. **欄位可用性評估 (Feature Availability)**：判斷哪些欄位因缺失值過多而不適合直接用於模型訓練，例如細分的病毒類別標籤。
3. **直觀視覺化 (Bar Plotting)**：將統計結果以長條圖呈現，快速對比各欄位間的資料缺失嚴重程度。

```python
missing_vals = train_df.isnull().sum()
missing_vals.plot(kind = 'bar')
```
<img width="600" height="655" alt="6" src="https://github.com/user-attachments/assets/dff19cba-1d64-4097-a6c4-a25316e6f8a3" />

---

## 步驟七：處理與驗證缺失值 (Handling & Verifying Missing Values)

在此步驟中，我們嘗試過濾無效數據，並重新檢查資料表的缺失值狀態，以確保後續資料處理的品質。

### 檢查重點：

1. **清除全空列 (Dropping Empty Rows)**：使用 `dropna(how = 'all')` 嘗試移除那些所有欄位皆為缺失值的資料列，避免無意義的空數據進入訓練流程。
2. **操作驗證 (Post-cleaning Verification)**：再次執行 `isnull().sum()`，觀察在執行清理操作後，各個欄位的缺失值數量是否有變化。
3. **資料一致性 (Data Consistency)**：確認核心欄位（如影像名稱與主要標籤）是否仍保有足夠的資料量，並評估是否需要進一步針對特定欄位進行填補或刪除。

```python
train_df.dropna(how = 'all')
train_df.isnull().sum()

```
<img width="275" height="322" alt="7" src="https://github.com/user-attachments/assets/940c2da3-6d61-4a8e-a079-a0cdbd14ee33" />

---

## 步驟八：缺失值填補與最終確認 (Missing Value Imputation & Final Check)

為了保持資料集的完整性並避免在後續處理中發生錯誤，我們將剩餘的缺失值進行統一填補。

### 檢查重點：

1. **統一標記 (Uniform Labeling)**：使用 `fillna('unknown', inplace=True)` 將所有缺失的數值（NaN）替換為字串 `'unknown'`。這對於像 `Label_2_Virus_category` 這種非必填欄位特別有用，能將缺失資訊轉化為一個明確的類別。
2. **原地更新 (In-place Update)**：透過 `inplace=True` 參數直接修改原始的 `train_df` 記憶體空間，確保變更被永久儲存而不需要額外的賦值動作。
3. **歸零驗證 (Final Verification)**：最後一次執行 `isnull().sum()`，目標是確認所有欄位的缺失值計數皆已歸零，確保資料集已處於「乾淨」且可供模型讀取的狀態。

```python
train_df.fillna('unknown', inplace=True)
train_df.isnull().sum()
```
<img width="344" height="318" alt="8" src="https://github.com/user-attachments/assets/f011a0cc-a746-4307-86e1-10268cc8b34b" />

---

## 步驟九：資料集劃分與驗證 (Dataset Splitting & Validation)

在此步驟中，我們根據資料集中預設的 `Dataset_type` 欄位，將原始資料表正式拆分為訓練集（Train Set）與測試集（Test Set），並進行完整性檢查。

### 檢查重點：

1. **子集過濾 (Subset Filtering)**：透過條件篩選，分別將標記為 `'TRAIN'` 與 `'TEST'` 的數據提取到獨立的 DataFrame 中，以便後續進行模型訓練與性能評估。
2. **斷言校驗 (Integrity Check)**：利用 `assert` 陳述式驗證兩個子集的總列數是否完全等於原始資料表的總列數，確保資料在分割過程中沒有遺漏或重疊。
3. **規模確認 (Shape Inspection)**：輸出訓練與測試數據的維度（Shape），讓我們對可用於學習的樣本數量有精確的掌握。
4. **測試樣本抽樣 (Random Sampling)**：透過 `sample(10)` 隨機抽取 10 筆測試資料，初步觀察測試集的標籤分佈與資料品質。

```python
train_data = train_df[train_df['Dataset_type'] == 'TRAIN']
test_data = train_df[train_df['Dataset_type'] == 'TEST']
assert train_data.shape[0] + test_data.shape[0] == train_df.shape[0]
print(f"Shape of train data : {train_data.shape}")
print(f"Shape of test data : {test_data.shape}")
test_data.sample(10)

```
<img width="959" height="512" alt="9" src="https://github.com/user-attachments/assets/d89124f8-fd91-425a-a252-7ab0b330ea7a" />

---

## 步驟十：標籤類別分佈統計 (Label Distribution Analysis)

透過執行 `value_counts()` 函式，我們統計了資料集中各個層級標籤的樣本數量。這對於識別類別不平衡（Class Imbalance）問題至關重要，因為不平衡的數據分佈可能會影響模型的預測偏好。

### 檢查重點：

1. **第一層類別分佈 (Primary Category Counts)**：統計 `Label_1_Virus_category`，了解病毒性、細菌性肺炎與正常樣本的基礎比例。
2. **第二層細分統計 (Secondary Category Counts)**：分析 `Label_2_Virus_category` 中的具體病原體佔比。在此處我們可以看到先前填補的 `'unknown'` 類別與具體疾病（如 COVID-19）的樣本數量。
3. **訓練策略參考 (Training Strategy Reference)**：根據統計結果評估是否需要在訓練時加入類別權重（Class Weights）或進行過取樣（Oversampling），以確保模型不會過度偏向樣本數較多的類別。

```python
print((train_df['Label_1_Virus_category']).value_counts())
print('--------------------------')
print((train_df['Label_2_Virus_category']).value_counts())

```
<img width="484" height="328" alt="10" src="https://github.com/user-attachments/assets/5b5260f1-8731-4c3c-8973-49e79460e549" />

---

## 步驟十一：影像檔案路徑配置與抽樣檢查 (Image Path Configuration & Sampling)

在此步驟中，我們定義了訓練集與測試集影像的存放路徑，並透過檔案系統遍歷來擷取部分樣本的路徑，以確認圖檔位置是否正確。

### 檢查重點：

1. **路徑變數定義 (Directory Definition)**：設定 `train_img_dir` 與 `test_img_dir`，將其指向解壓後的影像資料夾，建立程式碼與實體檔案之間的連結。
2. **檔案系統遍歷 (File System Traversal)**：利用 `os.walk` 掃描資料夾，並提取出檔案列表。透過 `map` 與 `lambda` 函式將檔名轉換為完整的絕對路徑。
3. **路徑有效性驗證 (Path Validation)**：輸出第一張訓練影像的完整路徑並檢查樣本清單長度。這能確保先前的解壓縮路徑正確無誤，且程式具備存取該目錄的權限。

```python
test_img_dir = '/content/chest_xray/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
train_img_dir = '/content/chest_xray/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'

sample_train_images = list(os.walk(train_img_dir))[0][2][:8]
sample_train_images = list(map(lambda x: os.path.join(train_img_dir, x), sample_train_images))

sample_test_images = list(os.walk(test_img_dir))[0][2][:8]
sample_test_images = list(map(lambda x: os.path.join(test_img_dir, x), sample_test_images))

print("第一張訓練片路徑:", sample_train_images[0])
print("訓練片數量:", len(sample_train_images))
```
<img width="1005" height="272" alt="11" src="https://github.com/user-attachments/assets/1bcf6da0-3f84-4c9d-a89d-79dddebc9bbb" />

---

## 步驟十二：訓練影像樣本視覺化 (Training Image Visualization)

在此步驟中，我們從訓練集中隨機選取部分影像路徑並將其實際繪製出來，這有助於我們直觀地觀察原始數據的特徵、品質以及 X 光片的成像狀況。

### 檢查重點：

1. **繪製圖像格柵 (Plotting Image Grid)**：使用 `plt.subplot` 建立一個  的佈局，以便在同一個畫布上同時查看 8 張訓練樣本影像。
2. **影像讀取與呈現 (Image Loading & Rendering)**：利用 `PIL` 庫開啟先前篩選出的影像路徑，並透過 Matplotlib 將其轉化為可視化圖表。
3. **X 光專用配色 (X-ray Optimized Colormap)**：在 `imshow` 中採用 `plt.cm.bone` 色圖進行顯示，這種配色模擬了傳統 X 光片的視覺效果，能更清晰地呈現骨骼與組織的對比。
4. **佈局自動最佳化 (Layout Optimization)**：透過 `tight_layout()` 自動調整子圖之間的間距，確保影像顯示整齊且不會互相重疊。

```python
plt.figure(figsize = (10,10))
for iterator, filename in enumerate(sample_train_images):
    image = PIL.Image.open(filename)
    plt.subplot(4,2,iterator+1)
    plt.imshow(image, cmap=plt.cm.bone)

plt.tight_layout()

```
<img width="855" height="641" alt="12" src="https://github.com/user-attachments/assets/bda58132-ffdf-4a55-a4c8-ba8aadc1a0ef" />

<img width="891" height="503" alt="12-1" src="https://github.com/user-attachments/assets/56c0049c-48ba-4107-a076-1d202c1b0845" />

---

## 步驟十三：細分標籤分佈視覺化 (Visualization of Fine-grained Label Distribution)

透過 Seaborn 的 `countplot` 函式，將訓練集中細分標籤（`Label_2_Virus_category`）的分佈情形以長條圖呈現。這能讓我們具象地看到資料集中各病原體樣本數量的顯著落差。

### 檢查重點：

1. **細節標籤分佈 (Detailed Label Distribution)**：呈現特定病原體（如 COVID-19、Streptococcus 等）與先前填補的 `'unknown'` 樣本之佔比。
2. **類別不平衡程度 (Degree of Class Imbalance)**：直觀判斷資料是否存在極端的長尾分佈。例如，COVID-19 樣本數若遠少於其他類別，則在後續模型訓練時可能需要透過損失函數權重調整來補償。
3. **圖表佈局優化 (Plot Scaling)**：設定較大的畫布尺寸 ，確保標籤名稱在橫軸上清晰可見，避免文字重疊導致閱讀困難。

```python
plt.figure(figsize=(15,10))
sns.countplot(train_data['Label_2_Virus_category']);

```
<img width="421" height="62" alt="13" src="https://github.com/user-attachments/assets/d263c09b-7905-4490-8e20-af33171e345e" />

<img width="925" height="548" alt="13-1" src="https://github.com/user-attachments/assets/4076ab8d-61a5-4961-a43f-bc2ccb84d6d7" />

---

## 步驟十四：COVID-19 影像與直方圖對比分析 (COVID-19 Image & Histogram Analysis)

在此步驟中，我們專注於觀察 COVID-19 樣本的影像特徵，並透過像素強度的直方圖（Histogram）來分析影像的亮度分佈與對比度特性。

### 檢查重點：

1. **特定標籤抽樣 (Targeted Sampling)**：從訓練集中篩選出所有標記為 `COVID-19` 的影像名稱，並擷取前 4 張樣本進行深入分析。
2. **影像與頻譜對照 (Image vs. Histogram)**：
* **影像側**：展示 X 光原始圖像，觀察病變區域（如磨玻璃影 GGO）的視覺特徵。
* **直方圖側**：利用 `image.ravel()` 將二維影像展開為一維，統計 0 到 255 的灰階像素分佈，分析影像是否有過曝、過暗或低對比度的情況。


3. **數據特徵工程參考 (Feature Engineering Insight)**：透過直方圖分佈，我們可以判斷是否需要進行影像增強（如直方圖等化或對比度調整），以協助模型更有效地提取細微特徵。
4. **視覺佈局優化 (Subplot Configuration)**：建立兩欄式的對比圖表，並關閉影像軸線（axis off）以保持介面簡潔。

```python
fig, ax = plt.subplots(4, 2, figsize=(15, 10))

covid_path = train_data[train_data['Label_2_Virus_category']=='COVID-19']['X_ray_image_name'].values

sample_covid_path = covid_path[:4]
sample_covid_path = list(map(lambda x: os.path.join(train_img_dir, x), sample_covid_path))

for row, file in enumerate(sample_covid_path):
    image = plt.imread(file)
    ax[row, 0].imshow(image, cmap=plt.cm.bone)
    ax[row, 1].hist(image.ravel(), 256, [0,256])
    ax[row, 0].axis('off')
    if row == 0:
        ax[row, 0].set_title('Images')
        ax[row, 1].set_title('Histograms')
fig.suptitle('Label 2 Virus Category = COVID-19', size=16)
plt.show()

```
<img width="772" height="337" alt="14" src="https://github.com/user-attachments/assets/5fb0cc14-9487-40cc-8a43-7621c877f8de" />

<img width="712" height="602" alt="14-1" src="https://github.com/user-attachments/assets/ae35fe82-dcbf-4d43-aee6-187595c3b8bf" />

---

## 步驟十五：正常樣本影像與直方圖對比分析 (Normal Image & Histogram Analysis)

在此步驟中，我們針對標記為 `Normal`（正常）的 X 光影像進行同樣的視覺化與像素頻譜分析。這能與前一步的 COVID-19 樣本形成對比，幫助我們辨識健康肺部與病變肺部在數位特徵上的差異。

### 檢查重點：

1. **基準樣本抽樣 (Baseline Sampling)**：從 `Label` 欄位篩選出正常樣本，並獲取其前 4 張影像路徑，建立分類任務的基準線（Baseline）。
2. **像素分佈對比 (Pixel Distribution Comparison)**：
* **視覺特徵**：觀察正常肺部影像的清晰度、肋骨輪廓以及肺野（Lung Fields）的透光度。
* **頻譜分析**：透過直方圖觀察像素強度分佈，通常正常影像的直方圖會與病變影像在特定灰階區間（如代表積水或浸潤的白色區域）表現出不同的峰值。


3. **資料品質一致性 (Data Quality Consistency)**：確認正常樣本的影像曝光與對比度是否與病變樣本一致，避免模型僅根據影像亮度而非醫學特徵來進行分類。

```python
fig, ax = plt.subplots(4, 2, figsize=(15, 10))


normal_path = train_data[train_data['Label']=='Normal']['X_ray_image_name'].values

sample_normal_path = normal_path[:4]
sample_normal_path = list(map(lambda x: os.path.join(train_img_dir, x), sample_normal_path))

for row, file in enumerate(sample_normal_path):
    image = plt.imread(file)
    ax[row, 0].imshow(image, cmap=plt.cm.bone)
    ax[row, 1].hist(image.ravel(), 256, [0,256])
    ax[row, 0].axis('off')
    if row == 0:
        ax[row, 0].set_title('Images')
        ax[row, 1].set_title('Histograms')
fig.suptitle('Label = NORMAL', size=16)
plt.show()
```
<img width="737" height="347" alt="15" src="https://github.com/user-attachments/assets/3d81c32f-32f6-430a-a896-9063ce571e7f" />

<img width="711" height="603" alt="15-1" src="https://github.com/user-attachments/assets/c5daa431-fadc-4cff-8d2b-e96fa5a5eb17" />

---

## 步驟十六：目標子集篩選與資料精簡 (Target Subset Filtering & Data Refinement)

在此步驟中，我們對訓練集進行了關鍵的過濾操作，旨在將任務聚焦於「正常影像」與「COVID-19 肺炎影像」的二分類或特定研究對比。

### 檢查重點：

1. **定義研究範圍 (Narrowing the Scope)**：透過邏輯篩選，我們僅保留了兩類數據：
* **Label 為 'Normal'** 的健康樣本。
* **Label 為 'Pnemonia' 且細分標籤為 'COVID-19'** 的病變樣本。


2. **排除雜訊 (Noise Reduction)**：此舉有效排除了其他類型的肺炎（如一般的細菌性或病毒性肺炎），這有助於模型學習更具特異性的 COVID-19 放射學特徵，提高模型在特定疾病識別上的精準度。
3. **資料重組 (Dataset Re-structuring)**：將篩選後的結果儲存於 `final_train_data` 中，這將作為後續影像生成器與模型訓練的最終數據源。

```python
final_train_data = train_data[(train_data['Label'] == 'Normal') | 
                              ((train_data['Label'] == 'Pnemonia') &
                               (train_data['Label_2_Virus_category'] == 'COVID-19'))]

```
<img width="999" height="89" alt="16" src="https://github.com/user-attachments/assets/86e5d274-ac57-4557-804c-e5b8beddfb85" />

---

## 步驟十七：標籤編碼與類別映射 (Label Encoding & Class Mapping)

為了讓深度學習模型能夠處理類別資料，我們需要將原始的文字標籤（Normal/Pnemonia）轉換為模型可識別的二元分類格式（0 與 1）。

### 檢查重點：

1. **類別命名 (Semantic Labeling)**：新增 `class` 欄位，將 'Normal' 映射為 `'negative'`，並將所有肺炎（包含篩選後的 COVID-19）映射為 `'positive'`。這有助於在後續繪製混淆矩陣或生成報告時更具易讀性。
2. **數值化轉換 (Numerical Encoding)**：建立 `target` 欄位作為模型的真實標籤（Ground Truth）。
* **0**：代表健康（Negative）。
* **1**：代表患病（Positive）。


3. **一致性處理 (Consistency Across Sets)**：同步對 `final_train_data` 與 `test_data` 執行相同的轉換邏輯，確保訓練與評估階段的標準一致。

```python
final_train_data['class'] = final_train_data.Label.apply(lambda x: 'negative' if x=='Normal' else 'positive')
test_data['class'] = test_data.Label.apply(lambda x: 'negative' if x=='Normal' else 'positive')

final_train_data['target'] = final_train_data.Label.apply(lambda x: 0 if x=='Normal' else 1)
test_data['target'] = test_data.Label.apply(lambda x: 0 if x=='Normal' else 1)

```
<img width="1004" height="128" alt="17" src="https://github.com/user-attachments/assets/6cb86aaf-5719-4da0-a5df-2ff7426a106f" />

---

## 步驟十八：選取核心欄位與精簡資料表 (Feature Selection & Table Refinement)

在此步驟中，我們對資料表進行了最終的「瘦身」，僅保留與模型訓練及評估直接相關的欄位。這不僅能減少記憶體佔用，更能讓後續將資料餵入影像生成器（Data Generator）時的邏輯更加簡潔。

### 檢查重點：

1. **欄位精簡 (Column Pruning)**：從原始繁雜的 Metadata 中提取出關鍵資訊。
* **`X_ray_image_name`**：作為連結磁碟圖片檔案的唯一索引。
* **`class`**：供影像生成器使用的字串類別標籤（'negative' / 'positive'）。
* **`target`**：數值化標籤，用於計算損失函數與評估指標。


2. **保留追蹤資訊 (Tracking Metadata)**：在訓練集中特別保留了 `Label_2_Virus_category`，這有助於在訓練後進行錯誤分析（Error Analysis），查看模型是否對特定病原體（如 COVID-19）有較高的誤判率。
3. **測試集標準化 (Test Set Standardization)**：確保 `final_test_data` 與訓練集結構一致，為後續的模型驗證與測試流程做好準備。

```python
final_train_data = final_train_data[['X_ray_image_name', 'class', 'target', 'Label_2_Virus_category']]
final_test_data = test_data[['X_ray_image_name', 'class', 'target']]

```
<img width="938" height="66" alt="18" src="https://github.com/user-attachments/assets/fae7d343-1151-4425-ba95-80ac547b5c69" />

---

## 步驟十九：測試集類別分佈統計 (Test Set Label Distribution Analysis)

在此步驟中，我們針對測試集（`test_data`）的原始標籤進行數量統計。這有助於我們在模型評估前，掌握測試數據的類別組成，確保測試基準的合理性。

### 檢查重點：

1. **驗證測試規模 (Verification of Test Scale)**：確認用於最後評估模型的測試樣本總數，並觀察「正常 (Normal)」與「肺炎 (Pnemonia)」樣本的實際比例。
2. **評估基準線 (Baseline Evaluation)**：了解類別分佈有助於解讀模型後續產出的準確率（Accuracy）。如果類別高度不平衡，我們可能需要更關注 F1-score 或 AUC 等指標。
3. **數據一致性檢查 (Consistency Check)**：確保測試集的標籤分佈符合預期，且沒有出現異常的數據偏差。

```python
test_data['Label'].value_counts()

```
<img width="360" height="257" alt="19" src="https://github.com/user-attachments/assets/126a7406-f161-48db-8a39-6c43096468d1" />

---

## 步驟二十：定義影像增強器與讀取函式 (Data Augmentation & Image Loading Function)

在此步驟中，我們設定了影像預處理的邏輯，包括資料增強（Data Augmentation）的參數以及一個自定義的影像讀取函式，確保影像能以統一的格式餵入模型。

### 檢查重點：

1. **資料增強配置 (Data Augmentation Setup)**：建立 `ImageDataGenerator` 並設定 `shear_range`（剪切變換）與 `zoom_range`（隨機縮放）。這些隨機變換能人為擴充訓練數據的多樣性，幫助模型學習更具魯棒性（Robust）的特徵，防止過擬合。
2. **格式標準化 (Standardized Loading)**：自定義 `read_img` 函式，確保所有影像在讀取時都會調整為統一的尺寸 (`target_size`)，這對於卷積神經網路的輸入要求至關重要。
3. **像素歸一化 (Pixel Normalization)**：在函式中將像素值除以 255，將範圍從  縮放到 。這有助於梯度下降算法更快地收斂。
4. **陣列轉換 (Array Conversion)**：使用 `img_to_array` 將 PIL 影像物件轉換為 NumPy 陣列，以便進行後續的數值運算。

```python
datagen =  ImageDataGenerator(
  shear_range=0.2,
  zoom_range=0.2,
)

def read_img(filename, size, path):
    img = image.load_img(os.path.join(path, filename), target_size=size)
    #convert image to array
    img = image.img_to_array(img) / 255
    return img

```
<img width="713" height="237" alt="20" src="https://github.com/user-attachments/assets/b73baad7-50d7-4c1d-990a-28309c687bdb" />

---

## 步驟二十一：影像增強效果測試與驗證 (Data Augmentation Testing & Verification)

在正式訓練模型之前，我們實際應用 `ImageDataGenerator` 對單張影像進行變換。這能讓我們確認增強參數（如剪切與縮放）是否設定得當，並確保生成的影像仍保有可辨識的醫學特徵。

### 檢查重點：

1. **單一樣本預覽 (Single Sample Preview)**：讀取訓練集的第一張影像 `samp_img`，並將其調整為  的標準尺寸。
2. **維度擴充 (Dimension Expansion)**：透過 `tf.expand_dims(samp_img, 0)` 將單張影像包裝成批次格式（Batch），以符合 `datagen.flow` 的輸入要求（需為 4D 張量）。
3. **隨機變換展示 (Random Transformation Display)**：利用迴圈生成並顯示 9 張不同的增強影像。每一張影像都會隨機應用不同程度的 `shear`（剪切）與 `zoom`（縮放），模擬拍攝角度與距離的細微差異。
4. **魯棒性確認 (Robustness Check)**：觀察增強後的影像是否出現過度扭曲。理想的資料增強應在增加資料多樣性的同時，不破壞 X 光片原有的病理資訊。

```python
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
)

def read_img(filename, size, path):
    full_path = os.path.join(path, filename)
    img = image.load_img(full_path, target_size=size)
    img = image.img_to_array(img) / 255
    return img

samp_img = read_img(final_train_data['X_ray_image_name'][0],
                    (255, 255),
                    train_img_dir)

plt.figure(figsize=(10,10))
plt.suptitle('Data Augmentation', fontsize=28)

i = 0
for batch in datagen.flow(tf.expand_dims(samp_img, 0), batch_size=1):
    plt.subplot(3, 3, i+1)
    plt.grid(False)
    plt.imshow(batch[0])
    
    if i == 8:
        break
    i += 1
    
plt.show()

```
<img width="652" height="729" alt="21" src="https://github.com/user-attachments/assets/0a0c3140-8708-4f8a-8f10-be060ab04950" />

<img width="587" height="588" alt="21-1" src="https://github.com/user-attachments/assets/a28025fb-bb1c-44cd-bc7f-9ffd944fc1a3" />


---

## 步驟二十二：針對 COVID-19 樣本進行資料增強擴充 (Targeted Data Augmentation for COVID-19)

由於 COVID-19 的原始影像數量較少，為了緩解類別不平衡（Class Imbalance）問題並提高模型的泛化能力，我們針對該類別進行了特定的大規模資料增強。

### 檢查重點：

1. **特定類別提取 (Class Targeting)**：從篩選後的資料中單獨提取出 `COVID-19` 的樣本路徑，作為本次增強處理的核心對象。
2. **倍數擴充策略 (Multiplication Strategy)**：定義 `augment` 函式，對每一張原始 COVID-19 影像透過 `datagen.flow` 產生 20 張經過變換（如旋轉、縮放、剪切）的新影像。這能將少數類別的樣本規模擴大約 20 倍。
3. **進度追蹤 (Progress Monitoring)**：引入 `tqdm` 庫建立進度條，讓我們能即時監控大規模影像處理的執行進度與預估完成時間。
4. **記憶體內儲存 (In-memory Storage)**：將產生的增強影像轉化為 NumPy 陣列並存入 `with_corona_augmented` 清單中，方便後續與其他訓練數據進行合併。

```python
from tqdm.notebook import tqdm  
import numpy as np


corona_df = final_train_data[final_train_data['Label_2_Virus_category'] == 'COVID-19']
with_corona_augmented = []


def augment(name):
    
    img = read_img(name, (255, 255), train_img_dir) 
    
    i = 0
    for batch in datagen.flow(tf.expand_dims(img, 0), batch_size=1):
        
        augmented_img = batch[0] 
        with_corona_augmented.append(augmented_img)
        
        if i == 20: 
            break
        i += 1

print(f"開始增強 COVID-19 圖片，預計處理 {len(corona_df)} 張原圖...")
for img_name in tqdm(corona_df['X_ray_image_name']):
    augment(img_name)

print(f"增強完成！總共產生了 {len(with_corona_augmented)} 張圖片。")

```

<img width="678" height="605" alt="22" src="https://github.com/user-attachments/assets/26c7ecf9-5ff4-4662-8a99-dcf4c1888f7d" />

---

## 步驟二十三：全量影像讀取與陣列轉換 (Full Image Loading & Array Conversion)

在此步驟中，我們將所有篩選後的訓練與測試影像從磁碟讀取到記憶體中，並利用先前定義的 `read_img` 函式將其全數轉換為數值陣列格式。

### 檢查重點：

1. **批次預處理 (Batch Preprocessing)**：透過迴圈與 `read_img` 函式，將每一張影像自動進行尺寸調整（）與歸一化處理（除以 255），確保進入模型的特徵尺度一致。
2. **進度可視化 (Progress Visualization)**：使用 `tqdm` 追蹤讀取進度。由於處理上千張高解析度醫學影像涉及大量的磁碟 I/O 與計算，進度條能幫助我們掌握處理時間。
3. **記憶體整合 (Memory Integration)**：將所有處理後的影像陣列存入 `train_arrays` 與 `test_arrays` 清單。這是將非結構化的影像檔案轉化為結構化張量（Tensor）的關鍵步驟。
4. **資料量驗證 (Data Quantity Validation)**：讀取完成後輸出最終張數，確保所有在 CSV 中篩選出的樣本皆已成功轉換，沒有出現檔案損毀或路徑錯誤的情形。

```python
from tqdm.notebook import tqdm
import numpy as np


train_arrays = []
print("正在讀取訓練集圖片...")
for img_name in tqdm(final_train_data['X_ray_image_name']):
    
    img_array = read_img(img_name, (255, 255), train_img_dir)
    train_arrays.append(img_array)

test_arrays = []
print("正在讀取測試集圖片...")
for img_name in tqdm(final_test_data['X_ray_image_name']):
   
    img_array = read_img(img_name, (255, 255), test_img_dir)
    test_arrays.append(img_array)


print(f"訓練集讀取完成，共 {len(train_arrays)} 張")
print(f"測試集讀取完成，共 {len(test_arrays)} 張")

```
<img width="586" height="554" alt="23" src="https://github.com/user-attachments/assets/90394045-c7db-49fc-9f1a-baa9be157279" />


---

## 步驟二十四：標籤合併與最終訓練目標設定 (Label Concatenation & Final Training Target Setup)

在此步驟中，我們將原始訓練集的標籤與增強後的 COVID-19 標籤進行合併。這是構建最終訓練數據集的關鍵步驟，確保每個影像陣列都有其對應的真實標籤（Ground Truth）。

### 檢查重點：

1. **資料對齊 (Data Alignment)**：我們將原始資料的 `target` 數值（包含 Normal 與原始 COVID-19 樣本）與增強樣本的標籤進行串接。
2. **標籤分配 (Label Assignment)**：
* 原始標籤：直接取自 `final_train_data['target']`。
* 增強標籤：由於 `with_corona_augmented` 清單中全是 COVID-19 樣本，因此使用 `np.ones` 統一賦予標籤值 `1`（Positive）。


3. **資料型態統一 (Dtype Standardization)**：確保所有標籤皆使用 `np.int64` 型態，以符合深度學習框架對於分類標籤的格式要求。
4. **準備過取樣平衡 (Oversampling Completion)**：透過此操作，我們正式在標籤層面完成了「過取樣」，使模型訓練時能接觸到足夠多的正向（COVID-19）樣本。

```python
y_train = np.concatenate((np.int64(final_train_data['target'].values), np.ones(len(with_corona_augmented), dtype=np.int64)))

```
<img width="946" height="35" alt="24" src="https://github.com/user-attachments/assets/7c598353-1785-4e2a-abba-99352605b84a" />

---
## 步驟二十五：構建高效能 TensorFlow 資料管道 (Building TensorFlow Data Pipeline)

在此步驟中，我們將記憶體中的 NumPy 陣列轉換為 TensorFlow 的張量（Tensor）格式，並利用 `tf.data.Dataset` 封裝，這是為了利用 TensorFlow 的優化機制，在模型訓練時達到更快的讀取與處理效率。

### 檢查重點：

1. **張量轉換 (Tensor Conversion)**：
* **影像合併**：將原始影像陣列 `train_arrays` 與資料增強後的 `with_corona_augmented` 合併後轉換為訓練張量。
* **類型轉換**：透過 `tf.convert_to_tensor` 將數據正式從 NumPy 格式移轉至 TensorFlow 運算圖中。


2. **資料管道封裝 (Dataset Encapsulation)**：
* 使用 `from_tensor_slices` 將影像張量與對應標籤（`y_train_tensor`）成對綁定。
* 這種結構允許後續輕鬆加入 `.shuffle()` (打散資料) 與 `.batch()` (批次處理) 等高效操作。


3. **訓練與測試集分離**：確保 `train_dataset` 與 `test_dataset` 格式一致，這對於模型在訓練期間進行即時驗證至關重要。

```python
train_tensors = tf.convert_to_tensor(np.concatenate((np.array(train_arrays), np.array(with_corona_augmented))))
test_tensors  = tf.convert_to_tensor(np.array(test_arrays))
y_train_tensor = tf.convert_to_tensor(y_train)
y_test_tensor = tf.convert_to_tensor(final_test_data['target'].values)

train_dataset = tf.data.Dataset.from_tensor_slices((train_tensors, y_train_tensor))
test_dataset = tf.data.Dataset.from_tensor_slices((test_tensors, y_test_tensor))

```
<img width="863" height="149" alt="25" src="https://github.com/user-attachments/assets/d57e7ae3-8ba9-412a-b984-bd77fde0aba0" />

---

## 步驟二十六：資料管道最終驗證 (Data Pipeline Verification)

在正式將數據餵入模型之前，從 `tf.data.Dataset` 物件中提取一個樣本進行視覺化是標準的「冒煙測試」（Smoke Test）。這能確保影像張量與標籤在經過轉換與合併後，依然保持正確的結構與數值範圍。

### 檢查重點：

1. **資料迭代測試 (`.take(1)`)**：從資料集中抓取一個批次（或單個樣本），確認資料管道（Pipeline）運作順暢，沒有出現維度不匹配或類型錯誤。
2. **影像還原度確認**：透過 `plt.imshow(i)` 檢查影像是否正確顯示。因為我們之前進行了歸一化（除以 255），此時的像素值應在  之間，Matplotlib 能自動正確渲染。
3. **標籤對應檢查**：雖然程式碼中未印出標籤 `l`，但您可以透過觀察影像（如是否有肺炎特徵）來對照標籤值，確保「影像-標籤」的配對沒有在合併過程中產生錯位。

```python
for i, l in train_dataset.take(1):
    plt.imshow(i);

```
<img width="494" height="480" alt="26" src="https://github.com/user-attachments/assets/000598af-3f71-4bfd-ad71-412efd16e024" />

---

## 步驟二十七：批次處理與數據洗牌 (Batching & Shuffling)

這是模型訓練前的最後一個準備步驟。我們將原始的資料集切分成多個小批次（Batches），並對訓練集進行隨機洗牌，這對於隨機梯度下降（SGD）等優化演算法的收斂至關重要。

### 檢查重點：

1. **隨機洗牌 (`shuffle`)**：設定 `BUFFER = 1000`。這表示系統會從資料集中隨機選取 1000 個樣本放入緩衝區，並從中隨機抽取。這能確保模型在訓練時不會因為樣本的排列順序（例如先全是 Normal，後全是 COVID）而產生過擬合或偏見。
2. **批次大小 (`batch_size`)**：設定為 `16`。這是一個平衡記憶體佔用與梯度估計穩定性的常見數值。
3. **維度驗證 (Shape Validation)**：
* **訓練集批次**：應顯示為 `(16, 255, 255, 3)`，代表每批包含 16 張  的 RGB 影像。
* **測試集批次**：同樣進行驗證，確保模型評估時的輸入維度一致。


4. **效能優化**：透過 `tf.data` 的批次處理，TensorFlow 能在 GPU 運算時預先讀取下一個批次，大幅提升訓練速度。

```python
BATCH_SIZE = 16
BUFFER = 1000

train_batches = train_dataset.shuffle(BUFFER).batch(BATCH_SIZE)
test_batches = test_dataset.batch(BATCH_SIZE)

for i,l in train_batches.take(1):
    print('Train Shape per Batch: ',i.shape);
for i,l in test_batches.take(1):
    print('Test Shape per Batch: ',i.shape);

```
<img width="514" height="252" alt="27" src="https://github.com/user-attachments/assets/151b9de7-3269-4e37-bebc-04a84557729a" />

---

## 步驟二十八：遷移學習——載入 ResNet50 預訓練模型 (Transfer Learning with ResNet50)

在此步驟中，我們導入了強大的 **ResNet50** 卷積神經網路。透過「遷移學習（Transfer Learning）」，我們能利用該模型在大型數據集（ImageNet）上學習到的豐富特徵提取能力，來處理特定的醫學影像任務。

### 檢查重點：

1. **基礎模型選擇 (Base Model Selection)**：載入 `ResNet50`。這是一種具有 50 層深度的殘差網路（Residual Network），能有效解決深層網路中的梯度消失問題。
2. **輸入維度匹配 (Input Shape)**：設定 `input_shape=(255, 255, 3)`，確保模型的第一層能正確接收我們預處理後的 X 光影像數據。
3. **特徵提取器模式 (`include_top=False`)**：
* 我們去除了 ResNet50 原本用於 1000 類分類的全連接層（Top layer）。
* 此舉將模型轉化為一個強大的「特徵提取器」，輸出的將是影像的高維特徵圖（Feature Maps）。


4. **凍結權重 (`trainable = False`)**：
* **關鍵策略**：我們暫時鎖定模型的所有權重。這樣在訓練初期，預訓練的特徵提取層不會被破壞，只有我們後續添加的「自定義分類層」會進行學習。
* **優點**：大幅減少運算量，並防止在小數據集上發生嚴重的過擬合。



```python
INPUT_SHAPE = (255,255,3) 

base_model = tf.keras.applications.ResNet50(input_shape= INPUT_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# 凍結基礎模型，防止預訓練權重在初期被大幅更動
base_model.trainable = False

```
<img width="1045" height="219" alt="28" src="https://github.com/user-attachments/assets/f03e32b2-7e29-4a48-b528-f16e074c3405" />

---

## 步驟二十九：驗證特徵提取器的輸出維度 (Validating Feature Extractor Output)

在建構最終分類層之前，我們需要確認 **ResNet50** 在移除頂層（Top layer）後的輸出形狀。這有助於我們決定接下來該使用哪種池化（Pooling）或平坦化（Flattening）層。

### 檢查重點：

1. **張量流動測試 (Forward Pass Test)**：透過 `base_model(i)` 對一個批次的影像進行前向傳播。這不涉及梯度計算，純粹是為了檢查數據經過卷積基座（Convolutional Base）後的變化。
2. **輸出維度分析 (Output Shape)**：
* 預期輸出通常為 `(16, 8, 8, 2048)`。
* **16**：批次大小（Batch Size）。
* **8 x 8**：特徵圖的空間解析度（Spatial Resolution），影像從  經過多次下採樣縮小至此。
* **2048**：通道數（Channels/Filters），代表 ResNet50 提取出的抽象特徵數量。


3. **後續對接依據**：了解空間維度為  後，我們通常會接上 `GlobalAveragePooling2D()` 將其壓縮為 `(16, 2048)`，再送入全連接層進行二元分類。

```python
for i, l in train_batches.take(1):
    pass
print(base_model(i).shape)

```
<img width="359" height="93" alt="29" src="https://github.com/user-attachments/assets/87830ea3-ba00-4d37-b9db-0f736e857390" />

---
## 步驟三十：建構自定義分類模型與模型總覽 (Building Custom Classifier & Model Summary)

在此步驟中，我們將預訓練的 **ResNet50** 與自定義的分類層正式串接。透過這種「特徵提取 + 分類器」的結構，我們能將 ImageNet 的通用視覺知識轉化為針對 X 光影像的診斷能力。

### 核心架構解析：

1. **卷積基座 (Convolutional Base)**：將 `base_model`（ResNet50）作為第一層，負責從原始像素中提取複雜的空間特徵。
2. **全域平均池化 (Global Average Pooling 2D)**：將卷積層輸出的  特徵圖壓縮為一個  的向量。與傳統的 `Flatten` 相比，這能顯著減少參數數量並降低過擬合風險。
3. **隱藏層 (Fully Connected Layer)**：添加一個具有 128 個神經元的 `Dense` 層並使用 `ReLU` 激活函數，用於學習特徵之間的高階非線性組合。
4. **正規化 (Dropout)**：設定 `Dropout(0.2)`，在訓練期間隨機斷開 20% 的神經元連接。這是一種強大的正則化技術，迫使模型不依賴特定神經元，進而提高泛化能力。
5. **輸出層 (Output Layer)**：最後使用單個神經元與 `Sigmoid` 激活函數，輸出一個 0 到 1 之間的機率值，代表該影像屬於「Positive (COVID-19)」的信心程度。

### 模型摘要 (Model Summary)：

透過 `model.summary()`，您可以觀察到模型參數分為兩部分：

* **Trainable params**：僅包含您新增的 Dense 層參數。
* **Non-trainable params**：ResNet50 的預訓練參數（目前已被鎖定）。

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers 

model = Sequential()
model.add(base_model)

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128, activation='relu')) 
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

```
<img width="585" height="582" alt="30" src="https://github.com/user-attachments/assets/2ae1485a-839f-4f5b-ad6b-1432cbaa63ff" />

---

## 步驟三十一：配置訓練參數與早停機制 (Training Configuration & Early Stopping)

在開始執行訓練之前，我們需要定義模型如何學習（優化器）、如何衡量錯誤（損失函數），以及何時應該停止訓練以避免浪費運算資源或過度擬合。

### 核心配置解析：

1. **早停機制 (Early Stopping Callback)**：
* **`monitor='val_loss'`**：密切監控驗證集上的損失函數值。
* **`patience=2`**：如果驗證損失連續 **2 個 Epoch** 沒有下降，訓練將提前自動終止。這能有效防止模型在訓練集上過度鑽研（Overfitting）而喪失了對新資料的泛化能力。


2. **優化器 (Optimizer)**：
* 使用 **`adam`**。這是目前深度學習中最受歡迎的優化器之一，它能根據梯度的一階與二階矩動態調整學習率，通常能比傳統的 SGD 更快收斂。


3. **損失函數 (Loss Function)**：
* 選用 **`binary_crossentropy`**。由於我們的任務是二元分類（0 或 1），此函數會計算預測機率分佈與真實標籤之間的交叉熵，是此類問題的標準配置。


4. **評估指標 (Metrics)**：
* 監控 **`accuracy`**（準確率）。這讓我們在訓練過程中能直觀地看到模型預測正確樣本的百分比。



```python
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

```
<img width="587" height="125" alt="31" src="https://github.com/user-attachments/assets/a885b5cf-83ff-4782-937e-ec2932563685" />

---

## 步驟三十二：啟動模型訓練 (Model Training Execution)

這是整個專案最核心的階段。我們將資料批次餵入模型，讓 ResNet50 的特徵提取能力與自定義分類層結合，開始學習區分正常肺部與 COVID-19 肺炎。

### 訓練過程解析：

1. **數據迭代 (`fit`)**：模型將遍歷 `train_batches` 中的所有影像。由於我們之前進行了數據增強（Oversampling），模型現在有足夠的 COVID-19 樣本來學習其獨特的磨砂玻璃樣（Ground-glass opacities）等特徵。
2. **驗證機制 (`validation_data`)**：每個 Epoch 結束後，模型會立即在 `test_batches` 上進行測試。這能讓我們即時觀察模型對未見過數據的預測能力，並對比訓練集與測試集的準確率差異。
3. **動態終止 (`callbacks`)**：我們設定的 `EarlyStopping` 會在後台運作。如果訓練到第 5 輪時驗證損失不再下降，系統會自動在第 7 輪（）停止，保留最佳的模型狀態。
4. **效能預期**：在遷移學習初期，因為基礎層被凍結，訓練速度會非常快，且準確率通常會在前幾個 Epoch 迅速攀升。

```python
model.fit(train_batches, 
          epochs=10, 
          validation_data=test_batches, 
          callbacks=[callbacks])

```
<img width="1021" height="166" alt="32" src="https://github.com/user-attachments/assets/7e84522f-b57d-405b-ad4e-d9bc3176da27" />

---

## 步驟三十三：模型預測與二元門檻化 (Model Prediction & Binary Thresholding)

訓練完成後，我們將模型應用於完全未見過的測試資料集 `test_arrays`。這是評估模型臨床應用價值的關鍵環節，我們將模型輸出的「機率值」轉換為具體的「分類診斷」。

### 核心操作解析：

1. **機率預測 (`model.predict`)**：
* 模型對每張 X 光片會輸出一個介於  之間的數值。
* 接近 **0** 表示模型認為影像趨向於「Normal（陰性）」。
* 接近 **1** 表示模型認為影像趨向於「Positive（陽性，即 COVID-19）」。


2. **設定判定門檻 (Thresholding)**：
* 我們使用 `0.5` 作為分類標準： 判定為陽性，反之為陰性。
* `astype("int32")` 將布林值轉換為整數 **0** 與 **1**，以便與原始標籤進行對比。
```python
import numpy as np

predictions = model.predict(np.array(test_arrays))

pred = (predictions > 0.5).astype("int32")

```
<img width="467" height="154" alt="33" src="https://github.com/user-attachments/assets/a0b74bcc-6f61-49d5-8a51-b2ba17fcb077" />

---

## 步驟三十四：效能評估報告 (Classification Report Analysis)

透過 `classification_report`，我們不只看整體的「準確率（Accuracy）」，更能深入分析模型對於 COVID-19 陽性樣本的「捕捉能力（Recall）」以及「預測精準度（Precision）」。

### 指標解讀指南：

1. **精確率 (Precision)**：模型預測為陽性的樣本中，有多少真的是陽性？


2. **召回率 (Recall)**：所有的陽性病患中，模型成功抓出了多少人？在醫學診斷中，**高召回率**至關重要，因為我們不能漏掉任何潛在患者。


3. **F1-Score**：精確率與召回率的調和平均數。當資料不平衡（如 COVID 樣本較少）時，F1-Score 比單純的 Accuracy 更具參考價值。
4. **Flatten 操作**：由於 `pred` 是二維陣列（為了符合 Keras 輸出格式），我們使用 `.flatten()` 將其轉為一維，以便與測試標籤 `test_data['target']` 進行逐一比對。

```python
# classification report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(test_data['target'], pred.flatten()))

```
<img width="547" height="230" alt="34" src="https://github.com/user-attachments/assets/475d5c3c-afa1-4fce-bc67-6b4ed94969df" />

---

## 步驟三十五：混淆矩陣視覺化 (Confusion Matrix Visualization)

這是診斷模型表現最直觀的方式。混淆矩陣將預測結果與實際標籤進行交叉比對，讓我們能清楚看到模型是「精準捕捉」還是「嚴重誤判」。

### 矩陣區域解析：

1. **左上角 (True Negative, TN)**：實際為 Negative 且模型也預測為 Negative。代表正確排除正常樣本。
2. **右下角 (True Positive, TP)**：實際為 Positive 且模型也預測為 Positive。這是我們最希望看到的診斷成功案例。
3. **右上角 (False Positive, FP)**：**誤報**。實際正常卻被判為 COVID。這會導致不必要的恐慌與醫療資源浪費。
4. **左下角 (False Negative, FN)**：**漏報**。實際患病卻被判為正常。這在醫學上最危險，因為會導致患者錯失治療時機並造成社區傳播。

### 程式碼關鍵點：

* **`sns.heatmap`**：利用 Seaborn 繪製熱力圖，顏色深淺代表樣本數量的多寡。
* **`annot=True`**：在格子中直接顯示數字，讓我們能精確掌握分類張數。
* **`cmap='cividis'`**：使用對色覺障礙友好的色板，確保圖表易於閱讀。

```python
con_mat = confusion_matrix(test_data['target'], pred.flatten())
plt.figure(figsize = (10,10))
plt.title('CONFUSION MATRIX')
sns.heatmap(con_mat, cmap='cividis',
            yticklabels=['Negative', 'Positive'],
            xticklabels=['Negative', 'Positive'],
            annot=True);

```
<img width="505" height="144" alt="35" src="https://github.com/user-attachments/assets/205dfc11-a0d5-405d-8c41-5efc41aeee52" />

<img width="524" height="548" alt="35-1" src="https://github.com/user-attachments/assets/b80c23e6-fe7a-42a1-84f3-b637d6996edb" />

---

## 結論：模型表現分析與改進策略

### 1. 模型效能診斷 (Performance Diagnosis)

本次實作使用 ResNet50 遷移學習模型進行 COVID-19 影像分類。最終測試集準確率為 **62%**，然而透過混淆矩陣與分類報告發現，模型出現了**「多數類別偏誤 (Majority Class Bias)」**現象。

* **現象分析**：模型將所有測試樣本皆預測為 Positive (COVID-19)。雖然這使得陽性樣本的**召回率 (Recall)** 達到 1.0，但陰性樣本的識別能力為 0。
* **成因探討**：
* **資料不平衡**：測試集中陽性樣本佔比約 62.5%，模型在訓練初期陷入了預測多數類別以換取低損失值的「局部優化陷阱」。
* **特徵遷移落差**：ResNet50 的預訓練權重來自 ImageNet（一般物體），其特徵提取層對於醫學 X 光片的微細病理特徵（如毛玻璃樣浸潤影）不夠敏感。



### 2. 未來優化方向 (Future Enhancements)

若要提升模型在臨床上的實用性，我們未來可以採取以下改進措施：

* **模型微調 (Fine-tuning)**：解凍預訓練模型的高階卷積層，讓模型針對 X 光影像進行權重微調。
* **引入類別權重 (Class Weights)**：在訓練時賦予少數類別更高的權重損失，迫使模型關注特徵而非樣本數量。
* **調整判定門檻**：不侷限於 0.5 的門檻值，透過 ROC 曲線尋找最佳的平衡點。

---

## 實作心得：從開發過程中學到的核心能力

在這次 COVID-19 影像分類的專案實作中，我們獲得了以下幾點寶貴的經驗：

1. **完整資料流水線的建置**：
我們學會了如何從原始影像檔案出發，透過 `os` 進行路徑管理、使用 `ImageDataGenerator` 進行資料增強，並最終封裝成高效能的 `tf.data.Dataset`。這讓我們理解到，**「資料品質與預處理」**在深度學習中佔了 80% 的重要性。
2. **遷移學習的威力與限制**：
實作中我們體會到遷移學習（Transfer Learning）能快速搭建模型框架，但也深刻體認到，當預訓練領域（一般物體）與目標領域（醫學影像）差異巨大時，必須搭配適當的**微調 (Fine-tuning)** 策略才能發揮效果。
3. **評估指標的深度理解**：
過去我們往往只看「準確率 (Accuracy)」，但這次實作讓我明白在類別不平衡的資料集中，**混淆矩陣**與 **F1-Score** 才是揭露模型真實表現的關鍵工具。看到 Recall 1.0 但 Precision 不足的情況，讓我們更懂得從統計學角度解讀模型行為。

---



