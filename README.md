# PRG08 - Machine Vision & Neural Network

Dit is Bubble Pointer, een game die ik met  behulp van de Mediapipe API en ML5 heb gemaakt.

# Werking
Het project heeft drie verschillende pagina's.

- Game:
De plek waar je het spel kan spelen en je hoogste scores kan halen

- Training:
De plek waar je een ML% neural network kan trainen op basis van de Handlandmark data van Mediapipe. Zodra een model getrained is door de data in trainingData.json te zetten, laat de pagina zien of het model werkt en kun je de ruwe output bekijken van het model, naast een gesorteerde output.

 - Testing:
De plek waar de accuraatheid van het model berekend word en een Confusion Matrix te generen is op basis van de trainingData.json en de output van het model


# Known Issues

- Het model maakt soms een fout geeft soms 'nothing' aan terwijl het een duim of een pink is. Dit ligt aan de traingsdata.

# Live Omgeving

De live omgeving: [https://sijmenlokers.nl/bubble-pointer](https://sijmenlokers.nl/bubble-pointer)
