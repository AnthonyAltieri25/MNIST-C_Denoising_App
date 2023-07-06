FROM python:3.9-slim

#Sets our working directory
WORKDIR /app

#Gets code from app folder and requirement 
COPY app ./app
COPY requirements.txt ./

#Installs dependencies
RUN pip3 install -U pip && pip3 install -r requirements.txt

#Default port that streamlit uses
EXPOSE 8501

#Checks to make sure it is using port 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

#Runs the streamlit run command
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]