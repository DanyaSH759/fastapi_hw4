
def get_categories_vit(new_image_path):

    import json
    import torch
    from torchvision import transforms
    import torch.nn as nn
    import pandas as pd
    from PIL import Image
    import timm
    import torch.nn.functional as F


    class_to_idx = {"Bear": 0, "Brown bear": 1, "Bull": 2, "Camel": 3, "Canary": 4, "Cat": 5,
                     "Caterpillar": 6, "Cattle": 7, "Centipede": 8, "Cheetah": 9, "Chicken": 10, "Crab": 11,
                       "Crocodile": 12, "Deer": 13, "Dog": 14, "Duck": 15, "Eagle": 16, "Elephant": 17, "Fish": 18,
                         "Fox": 19, "Frog": 20, "Giraffe": 21, "Goat": 22, "Goldfish": 23, "Goose": 24, "Hamster": 25,
                           "Harbor seal": 26, "Hedgehog": 27, "Hippopotamus": 28, "Horse": 29, "Jaguar": 30, "Jellyfish": 31,
                             "Kangaroo": 32, "Koala": 33, "Ladybug": 34, "Leopard": 35, "Lion": 36, "Lizard": 37, "Lynx": 38,
                               "Magpie": 39, "Monkey": 40, "Moths and butterflies": 41, "Mouse": 42, "Mule": 43, "Ostrich": 44,
                                 "Otter": 45, "Owl": 46, "Panda": 47, "Parrot": 48, "Penguin": 49, "Pig": 50, "Polar bear": 51,
                                   "Rabbit": 52, "Raccoon": 53, "Raven": 54, "Red panda": 55, "Rhinoceros": 56, "Scorpion": 57,
                                     "Sea lion": 58, "Sea turtle": 59, "Seahorse": 60, "Shark": 61, "Sheep": 62, "Shrimp": 63,
                                       "Snail": 64, "Snake": 65, "Sparrow": 66, "Spider": 67, "Squid": 68, "Squirrel": 69,
                     "Starfish": 70, "Swan": 71, "Tick": 72, "Tiger": 73, "Tortoise": 74, "Turkey": 75, "Turtle": 76, "Whale": 77, 
                     "Woodpecker": 78, "Worm": 79, "Zebra": 80}


    # Определение трансформаций 
    transform_norm_new = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5018, 0.4925, 0.4460], # Наши подсчитанные значения mean и std для нормализации
                            std=[0.2339, 0.2276, 0.2402])
    ])


    # Load pre-trained VIT model
    model = timm.create_model('/fastapi_app/src/operations/DL_dicts/vit_base_patch32_clip_224.openai_ft_in1k', pretrained=True) # to check others models as well

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_classes = 81 #no_of_classes
    model.head = nn.Linear(model.head.in_features, num_classes, device = device)

    # Then load the state dictionary
    # get_vit_file() # download file if not exists
    model.load_state_dict(torch.load('/fastapi_app/src/operations/DL_dicts/vit_base_patch32_state_dict.pth', map_location=device))

   
    # Test the model on one image
    
    # Load image
    #image_path = 'extra images test/shiba-inu.webp' # test image path
    image_path = new_image_path # test image path
    image = Image.open(image_path).convert('RGB')

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size that model expects
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5018, 0.4925, 0.4460], std=[0.2339, 0.2276, 0.2402]),  # Normalize
    ])

    image = transform(image).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():  # Disable gradient tracking
        image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)

    # Get top categories
    top_num = 5  # Number of top categories you want
    top_prob, top_catid = torch.topk(probabilities, top_num)

    # Load category names
    # with open('/fastapi_app/src/operations/class_to_idx', 'r') as f:
    #     class_to_idx = json.load(f)

    # Get class names for prediction index
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Convert to Python data types and print
    top_prob = top_prob.cpu().numpy()[0]
    top_catid = top_catid.cpu().numpy()[0]

    predictions = []
    for i in range(top_num):
        predicted_class_name = idx_to_class[top_catid[i]]
        predicted_probability = top_prob[i]
        predictions.append({'Category ID': predicted_class_name, 'Probability': predicted_probability})

    df = pd.DataFrame(predictions)
    df_string = df.to_string(index=False)
    print(df_string)

    return df

def get_categories_rn(new_image_path):

    import json
    import torch
    from torchvision import transforms
    from PIL import Image
    import pandas as pd
    import torchvision.models as models
    import torch.nn as nn
    import torch.nn.functional as F

    # Load image
    image_path = new_image_path # test image path
    image = Image.open(image_path).convert('RGB')

    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size your model expects
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5018, 0.4925, 0.4460], std=[0.2339, 0.2276, 0.2402]),  # Normalize
    ])

    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Load category names
    with open('/fastapi_app/src/operations/class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)

    # First, recreate the model architecture
    resnet = models.resnet50(weights=True)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)

    #num_classes = len(class_to_idx)
    num_classes = 81
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes, device = device)

    # Then load the state dictionary
    # get_rn_file() # download file if not exists
    resnet.load_state_dict(torch.load('/fastapi_app/src/operations/DL_dicts/resnet50_state_dict.pth', map_location=device))

    # Move model to GPU
    resnet = resnet.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model to evaluation mode
    resnet.eval()

    with torch.no_grad():  # Disable gradient tracking
        image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = resnet(image)
        probabilities = F.softmax(outputs, dim=1)

    # Get top categories
    top_num = 5  # Number of top categories you want
    top_prob, top_catid = torch.topk(probabilities, top_num)

    # Get class names for prediction index
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Convert to Python data types and print
    top_prob = top_prob.cpu().numpy()[0]
    top_catid = top_catid.cpu().numpy()[0]

    predictions = []
    for i in range(top_num):
        predicted_class_name = idx_to_class[top_catid[i]]
        predicted_probability = top_prob[i]
        predictions.append({'Category ID': predicted_class_name, 'Probability': predicted_probability})

    df = pd.DataFrame(predictions)
    df_string = df.to_string(index=False)
    print(df_string)

    return df