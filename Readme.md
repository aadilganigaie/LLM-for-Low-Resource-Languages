Extending the Vocabulary of Large Language Models for Low-Resource Languages
In the realm of natural language processing (NLP), large language models (LLMs) have revolutionized various applications, from translation to text generation. However, most LLMs are English-centric, with limited support for low-resource languages like Kashmiri. This limitation arises primarily due to the scarcity of available corpora for these languages, making it challenging to train robust models.

One effective approach to address this issue is to extend the vocabulary of existing LLMs. By leveraging the emergent behavior of LLMs, we can enhance their performance on downstream tasks like translation for underrepresented languages. This article demonstrates how to extend the vocabulary of an LLM for Kashmiri and how this method can be adopted for other low-resource languages.

The Problem with Pretrained Tokenizers
Using pretrained tokenizers for underrepresented languages often leads to more token usage, which increases computational costs and power requirements. For instance, using a pretrained tokenizer for Kashmiri results in a higher number of tokens, leading to inefficiencies. To overcome this, extending the vocabulary of the existing LLM is beneficial.

Extending the Vocabulary
Step 1: Train a New Tokenizer
We start by training a new tokenizer specifically for Kashmiri using the SentencePiece library.

Step 2: Load and Test the New Tokenizer
After training the new tokenizer, we load it and test its performance on Kashmiri text:

Length of kashmiri text:  33
--------------
NEW TOKENIZER
--------------
Length of encoded IDs:  6
---
Compression ratio:  0.18
---
Encoded token IDs:  [385, 464, 1800, 12474, 1125, 826]
---
Decoded text:  اگر أس باقی اؠجِکؠشن سِسٹم وُچھو
--------------
MISTRAL TOKENIZER
--------------
Length of encoded IDs:  37
---
Compression ratio:  1.12
---
Encoded token IDs:  [1, 28705, 28915, 29461, 28947, 28705, 28915, 31865, 29008, 28705, 28983, 28915, 29115, 28975, 28705, 28915, 219, 163, 29156, 29421, 29130, 219, 163, 29083, 28955, 28705, 29008, 29421, 29008, 30634, 28954, 28705, 28962, 29587, 30066, 31052, 28962]
---
Decoded text:  <s> اگر أس باقی اؠجِکؠشن سِسٹم وُچھو

Step 3: Merge Tokenizers
Next, we merge the newly trained tokenizer with the existing Mistral tokenizer to ensure it includes non-English tokens:
Step 4: Check the Performance of the Extended Tokenizer
We then check the performance of the extended tokenizer on both Kashmiri and English texts:


Length of kashmiri text:  33
---
Length of english text:  23
--------------
EXTENDED MISTRAL TOKENIZER
--------------
Kashmiri Performance
---
Length of encoded IDs:  6
---
Compression ratio:  0.18
---
Encoded token IDs:  [32124, 32203, 33522, 44087, 32849, 32556]
---
Decoded text:  اگر أس باقی اؠجِکؠشن سِسٹم وُچھو
--------------
English Performance
---
Length of encoded IDs:  8
---
Compression ratio:  0.35
---
Encoded token IDs:  [315, 837, 4433, 28723, 1602, 460, 368, 28804]
---
Decoded text:  I am fine. How are you?
--------------
MISTRAL TOKENIZER
--------------
Length of encoded IDs:  9
---
Compression ratio:  0.39
---
Encoded token IDs:  [1, 315, 837, 4433, 28723, 1602, 460, 368, 28804]
---
Decoded text:  <s> I am fine. How are you?


Step 5: Convert SentencePiece Tokenizer to HuggingFace Tokenizer
Finally, we convert the SentencePiece tokenizer to a HuggingFace tokenizer and verify its performance:


Length of kashmiri text:  33
---
Length of english text:  23
--------------
EXTENDED MISTRAL TOKENIZER
--------------
Kashmiri Performance
---
Length of encoded IDs:  7
---
Compression ratio:  0.21
---
Encoded token IDs:  [1, 32124, 32203, 33522, 44087, 32849, 32556]
---
Decoded text:  <s> اگر أس باقی اؠجِکؠشن سِسٹم وُچھو
--------------
English Performance
---
Length of encoded IDs:  9
---
Compression ratio:  0.39
---
Encoded token IDs:  [1, 315, 837, 4433, 28723, 1602, 460, 368, 28804]
---
Decoded text:  <s> I am fine. How are you?
--------------
MISTRAL TOKENIZER
--------------
Length of encoded IDs:  9
---
Compression ratio:  0.39
---
Encoded token IDs:  [1, 28737, 837, 4433, 28723, 1602, 460, 368, 28804]
---
Decoded text:  <s>I am fine. How are you?


Step 6: Fine-Tune or Retrain the Model
Finally, we fine-tune or retrain the model using the extended tokenizer:

Compression Ratio and Token Usage: Before and After Extending the Vocabulary
One of the key challenges in working with low-resource languages is the inefficiency of pretrained tokenizers, which often result in a higher number of tokens and increased computational costs. By extending the vocabulary of an existing large language model (LLM), we can significantly improve the compression ratio and reduce the number of tokens used, making the model more efficient.
Before Extending the Vocabulary
When using the pretrained Mistral tokenizer on Kashmiri text, the token usage and compression ratio are as follows:

Kashmiri Text: "اگر أس باقی اؠجِکؠشن سِسٹم وُچھو"
Length of Kashmiri Text: 33 characters
Number of Encoded IDs: 37
Compression Ratio: 1.12
The pretrained tokenizer splits the Kashmiri text into a large number of tokens, leading to a compression ratio greater than 1, which is inefficient.

After Extending the Vocabulary
By training a new tokenizer specifically for Kashmiri and merging it with the existing Mistral tokenizer, we achieve the following results:

Kashmiri Text: "اگر أس باقی اؠجِکؠشن سِسٹم وُچھو"
Length of Kashmiri Text: 33 characters
Number of Encoded IDs: 6
Compression Ratio: 0.18
The extended tokenizer significantly reduces the number of tokens used, resulting in a much lower compression ratio. This improvement makes the model more efficient and cost-effective for processing Kashmiri text.

Comparison
Here is a side-by-side comparison of the token usage and compression ratio before and after extending the vocabulary:

Metric	Before Extending the Vocabulary	After Extending the Vocabulary
Number of Tokens Used	37	6
Compression Ratio	1.12	0.18
Detailed Output
Before Extending the Vocabulary:


Length of kashmiri text:  33
--------------
MISTRAL TOKENIZER
--------------
Length of encoded IDs:  37
---
Compression ratio:  1.12
---
Encoded token IDs:  [1, 28705, 28915, 29461, 28947, 28705, 28915, 31865, 29008, 28705, 28983, 28915, 29115, 28975, 28705, 28915, 219, 163, 29156, 29421, 29130, 219, 163, 29083, 28955, 28705, 29008, 29421, 29008, 30634, 28954, 28705, 28962, 29587, 30066, 31052, 28962]
---
Decoded text:  <s> اگر أس باقی اؠجِکؠشن سِسٹم وُچھو
After Extending the Vocabulary:


Length of kashmiri text:  33
--------------
NEW TOKENIZER
--------------
Length of encoded IDs:  6
---
Compression ratio:  0.18
---
Encoded token IDs:  [385, 464, 1800, 12474, 1125, 826]
---
Decoded text:  اگر أس باقی اؠجِکؠشن سِسٹم وُچھو
Conclusion
Extending the vocabulary of an existing LLM for low-resource languages like Kashmiri results in a significant reduction in the number of tokens used and a much lower compression ratio. This approach not only makes the model more efficient but also reduces computational costs, making it a viable solution for handling underrepresented languages. By adopting this method, we can enhance the performance of LLMs on various NLP tasks for low-resource languages, paving the way for more inclusive and robust language models.



