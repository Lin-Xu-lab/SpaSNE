FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir pandas==1.3.4 notebook scanpy scipy scikit-learn matplotlib

RUN make || g++ sptree.cpp spasne.cpp spasne_main.cpp -o spasne -O2

RUN python setup.py install

RUN cp spasne /usr/local/bin/

ENV PATH="/usr/local/bin:${PATH}"

CMD ["python", "spasne.py"]

