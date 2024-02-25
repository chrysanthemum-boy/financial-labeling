# financial-labeling
A system for automatic labeling of finance-related datasets.
## Django backend
### Run backend server
```
python manage.py migrate
python manage.py create_roles
python manage.py create_admin --noinput --username "admin" --email "admin@example.com" --password "password"
python manage.py runserver
```
### Run celery server
```
celery --app=config worker --loglevel=INFO --concurrency=1 -P eventlet
```

### Django URL reference

#### RESRful API
- http:localhost:8000/api-auth/login
- http:localhost:8000/api-auth/logout

#### Backend Service Monitoring
- http:localhost:8000/backend/health
