# 因子参数调整

这个项目只使用当前目录下的 `A股主表.csv`，用于给团队做可交互的参数调优和动态打分测试。

## 功能

- 从 `A股主表.csv` 生成应用专用 `parquet`
- 在网页里动态调整打分参数
- 对比调参结果与基准公式
- 输出 `Top N` 的简单历史表现
- 下载当前结果和参数配置

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

## 首版范围

- 当前只依赖 `A股主表.csv`
- 当前不做板块映射，不依赖 `包含度上服务器` 目录
- 当前主打个股层面的动态打分和回测对比
