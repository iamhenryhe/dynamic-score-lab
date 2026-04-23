# 因子参数调整

这个项目用于给团队做可交互的因子参数调整和板块总分测试。

## 功能

- 从 `A股主表.csv` 生成应用专用 `parquet`
- 使用 `板块映射表.csv` 计算股价包含度和板块容量
- 使用 `cbd/t-YYYY-MM-DD.csv` 读取板块传播度
- 在网页里动态调整包含度、容量、传播度和总分参数
- 对比调参结果与基准公式

## 本地运行

```bash
cd /Users/zijiehe/Desktop/动态打分
python3 scripts/build_app_dataset.py
streamlit run app/streamlit_app.py
```

## Docker 运行

```bash
cd /Users/zijiehe/Desktop/动态打分
docker build -t dynamic-score-lab .
docker run -p 8501:8501 dynamic-score-lab
```

## 数据依赖

- `A股主表.csv`
- `板块映射表.csv`
- `cbd/t-YYYY-MM-DD.csv`
- `data/derived/a_share_main_app.parquet`
