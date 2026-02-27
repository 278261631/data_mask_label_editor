## FITS 3D Viewer

一个用于浏览 FITS 图像并查看局部 3D 像素曲面的桌面程序。

### 功能

- **读取**：reference FITS（必需）与 aligned FITS（可选）
- **2D 浏览**：缩放、平移、reference/aligned 切换
- **3D 查看**：点击 2D 图像任意位置，显示该点附近局部区域的 3D surface
- **区域控制**：可调 3D 局部区域大小
- **配置数据目录**：默认 `D:\github\SiameseNetwork_fits_diff\data`

### 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 运行

```bash
python -m fits_3d_viewer
```

### 数据命名约定

程序会在数据目录中自动发现 FITS 主图，并尝试匹配：

- `xxx_1_reference.fits`（reference）
- `xxx_2_aligned.fits`（aligned，可选）

