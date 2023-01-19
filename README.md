# hand-written-Arabic-characters-recognition

Arabic characters recognition is the process of convert digital Arabic characters into machine-encoded text to recognize them. Character recognition consists of two types:
1. printed (OCR): OCR is the process of converting the document text, image of the document into machine text whether the text from scanned document and photos documents and furthermore. 
2. Handwritten recognition is the ability of computers to recognize the typed texts, characters on papers, texts on the signs, and furthermore.

Arabic character recognition is the most difficult classification task in the machine learning field, one of the problems that make it difficult that each character has different shapes in the writing.

In this model, we used Convolutional Neural Network (CNN) using dropout techniques to predict the characters. CNN is a deep learning algorithm that uses in terms of the visual image. CNN consists of three layers connected, these layers are; Input layer which stores row input data of the image processing in the network. Input layer accepted 3D input (width X height X depth) depth is for color channels in the color system (RGB). The second layer is feature-extraction layers; which consist of the Convolutional layer (Convolution), Pooling layer. Firstly, Convolutional layers (Convolution) transform the input data by using a patch of locally connecting neurons from the previous layer (input layer).
