from fastapi import APIRouter,  File, UploadFile



router = APIRouter(
    prefix="/operations",
    tags=["Operation"]
)


def redis_cache():
    return caches.get(CACHE_KEY)

# Проверка работоспасобности сервиса
@router.get("/ping")
async def get_ping_operations():
    return {"message": "Сервис активен"}


@router.post("/find_animal_vit")
async def upload_image_vit_operations(uploaded_img: UploadFile = File(...)):

    from operations.short_model import get_categories_vit

    file_location = f"/fastapi_app/src/operations/image_folder/{uploaded_img.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_img.file.read())

    test_return = get_categories_vit(f"/fastapi_app/src/operations/image_folder/{uploaded_img.filename}")

    return {"info": f"file '{uploaded_img.filename}' saved at '{file_location}. Animal list {test_return}'"}


@router.post("/find_animal_resnet")
async def upload_image_resnet_operations(uploaded_img: UploadFile = File(...)):

    from operations.short_model import get_categories_rn

    file_location = f"/fastapi_app/src/operations/image_folder/{uploaded_img.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_img.file.read())

    test_return = get_categories_rn(f"/fastapi_app/src/operations/image_folder/{uploaded_img.filename}")

    return {"info": f"file '{uploaded_img.filename}' saved at '{file_location}. Animal list {test_return}'"}
