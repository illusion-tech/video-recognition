FROM python:3.10.12

COPY ./requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 3001
COPY ./models /
COPY ./main.py /

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001"]
