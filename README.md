# chatbot
Chatbot implementation with LSTM on tensorflow2.0 and multi-gpu capability. This implementation is based on the keras tutorial https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html.

# Train model

```
from modules.train import ChatBotModel

model = ChatBotModel()

if __name__ == '__main__':
    model.train()
```

# Test model

```
from modules.chatbot import ChatBot

chatbot = ChatBot()

if __name__ == '__main__':
    chatbot.run("Hello, how are you?")
```