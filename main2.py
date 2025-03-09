from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import ast
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import random
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super(EmbeddingNet, self).__init__()
        # Use a pretrained ResNet18
        self.base_model = models.resnet18(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove original fc layer
        # Add new fully connected layers to produce embeddings.
        self.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        # L2 normalization (often beneficial for contrastive tasks)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

room_model = torch.load('room_model_837%_312am.pth', map_location=torch.device('cpu'), weights_only=False)
room_model.eval()

#room_model = torch.load('room_model_837%_312am.pth', map_location=torch.device('cpu'))
#room_model.eval()

img_size = 224  # typical input size for pretrained networks
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

furniture_df = pd.read_csv('furniture_embeddings.csv')
furniture_df['embedding'] = furniture_df['embedding'].apply(lambda x: torch.tensor(ast.literal_eval(x)))

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        img = Image.open(filepath).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            room_embedding = room_model(img_tensor)
        room_embedding = room_embedding.squeeze()  # remove batch dimension
    except Exception as e:
        print(f"Error processing room image: {e}")
        return redirect(url_for('index'))
    
    distances = []
    for idx, row in furniture_df.iterrows():
        furniture_embedding = row['embedding']
        distance = torch.norm(room_embedding - furniture_embedding)
        distances.append(distance.item())
    
    furniture_df['distance'] = distances
    top_matches = furniture_df.nsmallest(3, 'distance')
    
    results = []
    for idx, row in top_matches.iterrows():
        results.append({
            'image_url': row['Image URL'],
            'buy_link': row.get('Product Link', '#')  # fallback if buy_link not provided
        })
    
    return render_template('result3.html', filename=file.filename, results=results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
