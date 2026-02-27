## FITS Mask Label Editor

一个用于 **显示与编辑 FITS mask label** 的桌面程序（类似 labelme 的“画笔涂抹式”标注体验）。

### 功能（当前实现目标）

- **读取**：已对齐的 image FITS（16位） + mask FITS（16位整型 label map）
- **叠加显示**：mask 以半透明彩色覆盖在图像上
- **编辑**：选择标签（来自 `mask_codebook`），用画笔/橡皮在 mask 上涂抹
- **保存**：将修改后的 mask 写回 FITS（保留 16 位整型）
- **配置数据目录**：默认 `D:\github\SiameseNetwork_fits_diff\data`

### 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 运行

```bash
python -m mask_label_editor
```

### 数据与 codebook 约定（尽量兼容）

`mask_codebook` 建议使用 JSON（最稳）：

```json
{
  "labels": [
    { "code": 0, "name": "background", "color": "#000000" },
    { "code": 1, "name": "class1", "color": "#ff0000" }
  ]
}
```

也支持 CSV（列名任意，但需包含 code/name/color 或 r/g/b）：

- `code,name,color`
- `code,name,r,g,b`

若你目前的 `mask_codebook` 是 FITS（二进制表格 HDU），也会尝试自动识别常见列名。

