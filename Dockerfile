FROM public.ecr.aws/lambda/python:3.12

# Copie apenas o requirements.txt para aproveitar cache
COPY requirements.txt .

RUN python3.12 -m pip install --upgrade pip
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt
# Instale a versão CPU-only do PyTorch
RUN python3.12 -m pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
RUN python3.12 -m pip install --no-cache-dir pytorch-lightning==2.5.1.post0

COPY . .

CMD ["app.handler"]