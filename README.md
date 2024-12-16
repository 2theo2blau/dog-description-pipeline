# dog-description-pipeline
Pipeline for taking in photographs of dogs and other animals, captioning with an LLM, and outputting dog compliments through a TTS model.

**TTS Module Not Yet Implemented**

The application takes as input photographs of dogs (other subjects will likely work as well, but parts are designed specifically with dog photos in mind), performs object detection and crops the images to a fixed size, encodes them in base64, and then uses a series of requests to the Mistral API to first describe the dog in the photo and then come up with a creative compliment that is personal and specific to the dog in the photo.

```mermaid
graph TD
    A[Dog Photo] --> B[encode in base64]
    B --> C[Send to Pixtral for neutral caption]
    C --> D[Send caption to dog complimenting agent for personalized compliment]
```
