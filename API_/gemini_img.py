from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv('gemini_api_key')

from google import genai

#for image based tasks 
from PIL import Image

#Create gemini client
client = genai.Client(api_key= API_KEY)

#Load image
image = Image.open('./sample.jpg') #add the path of your image

#Generating content
result = client.models.generate_content(model= 'gemini-3.5-flash',
                                        contents = [
                                            'Describe everything you see in this image', 
                                            image] )
print(result.text)