FROM dregistry.0885c588-d112-482b-a98c-8e1f5a02d70c.k8s.civo.com/python:3.9-alpine

WORKDIR /src
COPY . .

RUN pip install pipenv==2023.7.23
RUN pipenv sync && pipenv install

EXPOSE 7777
CMD ["pipenv", "run", "flask"]