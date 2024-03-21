from fastapi import FastAPI

from auth.base_config import auth_backend, fastapi_users
from auth.schemas import UserRead, UserCreate

from operations.router import router as router_operation

import uvicorn

app = FastAPI(
    title="App"
)

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth",
    tags=["Auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["Auth"],
)

app.include_router(router_operation)

# # запуск приложения
# if __name__ == "__main__":
#     uvicorn.run("main:app", host = "127.0.0.1", port = 8888,  reload=True)