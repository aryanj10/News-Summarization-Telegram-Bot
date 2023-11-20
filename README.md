# News Summarization Telegram Bot

**Note: Unfortunately, this project is currently non-functional due to changes in the GNews API's policies.**

This bot is an news curation project that leverages Telegram's platform to deliver concise, reliable, and personalized news summaries to users. By utilizing advanced technologies such as sentence embedding with S-Bert models and integrating with the Google News API, this bot revolutionizes the way people consume news.

Users can effortlessly request summaries on specific news topics through the Telegram Chatbot interface. This Bot ensures the delivery of crisp, pertinent, and trustworthy news summaries sourced exclusively from verified and trusted sources. 

To ensure the utmost quality and relevance, we employed a sophisticated scoring system that evaluates content relevance, sentence novelty, and sentence position within articles. By applying a weighted average of these scores, the bot prioritizes the most informative and significant sentences in the summary. 

The robustness of the project is fueled by the utilization of the NLTK library for text preprocessing, the cutting-edge capabilities of sentence transformers (S-Bert) for vectorization, and the analytical power of NumPy for score calculation. The seamless integration with the Telegram API ensures swift and convenient delivery of these curated news summaries directly to users.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, NLTK, math, SBERT, Gnews, Telebot  

## Architecture

![image](https://github.com/aryanj10/News-Summarization-Telegram-Bot/assets/64386731/decf8002-6aa4-44b2-ac6e-73a5a314d83a)

## Sentence Scoring

![image](https://github.com/aryanj10/News-Summarization-Telegram-Bot/assets/64386731/bc3e3b70-e914-4c4a-9efc-e4fd8ab92f2b)

Suppose we have the following sentences:  
A man is eating food.  
A man is eating a piece of bread.  
The girl is carrying a baby.  

Content Relevance Score:  
Sentence 1 =  0.85,  
Sentence 2 =  0.84,  
Sentence 3 = 0.45  

Sentence Novelty Score:  
Sentence 1 =  0.15,  
Sentence 2 =  0.15,  
Sentence 3 =  0.99  

Sentence Position Score:  
'Sentence 1= 0.5,  
Sentence 2 = 0.5,  
Sentence 3 = 0.5  

Total Score:  
Sentence 1 = 0.645,  
Sentence 2 = 0.641,  
Sentence 3 = 0.57  

## Screenshots
<img width="190" alt="image" src="https://github.com/aryanj10/News-Summarization-Telegram-Bot/assets/64386731/65ab4e14-2ac7-4ca0-9ebc-4087c33bb43b">
<img width="172" alt="image" src="https://github.com/aryanj10/News-Summarization-Telegram-Bot/assets/64386731/de4e0dd2-1b7a-49f7-aa21-7944c4942280">
<img width="174" alt="image" src="https://github.com/aryanj10/News-Summarization-Telegram-Bot/assets/64386731/1c4b6062-8bfa-42a4-9299-66a4acf6c539">
<img width="174" alt="image" src="https://github.com/aryanj10/News-Summarization-Telegram-Bot/assets/64386731/d7474e4a-1ca0-442e-8ec5-08ab730ac8bc">
<img width="172" alt="image" src="https://github.com/aryanj10/News-Summarization-Telegram-Bot/assets/64386731/91cbfa89-0955-4d2f-9cb1-8a9d518d554b">




