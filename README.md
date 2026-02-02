
<p align="center" style="overflow: hidden; height: 200px; width: 300px; position: relative;">
  <img align="center" src="hisemotions_2026.png" style="position: absolute; bottom: -50px; width: 100%;" />
</p>

<h1 align="center">HISEMOTIONS 2026: Historical Text Based Emotion Detection in Early Modern Spanish Correspondence</h1>

- [Description of the Task](#description-of-the-task)
  - [Overview](#overview)

## Description of the Task

### Overview
  HISEMOTIONS 2026 proposes a shared task focused on Text-Based Emotion Detection (TBED) in Early Modern Spanish epistolary texts (16th-17th centuries). The task aims to foster the development and evaluation of NLP methods specifically adapted to historical Spanish language varieties, addressing the semantic and diachronic challenges of emotion and affective states expression in Early Modern Spanish correspondence.
  
While previous IberLEF shared tasks on sentiment and emotion analysis have focused on contemporary Spanish and modern text genres such as social media, reviews, and news, leaving the linguistic, orthographic, and semantic characteristics of Early Modern Spanish largely unexplored. Consequently, the transferability of models and resources developed for modern Spanish to historical language varieties remains unclear, particularly in domains where affective expression follows distinct semantic, cultural, and linguistic conventions. This shared task addresses this gap by evaluating emotion detection methods on Spanish historical correspondence. It seeks to answer core research questions concerning the generalisation of modern Spanish models to Early Modern Spanish, the impact of semantic change on emotion classification, and the effectiveness of domain- and diachrony-aware adaptation strategies compared to direct transfer. In doing so, the task aims to advance emotion detection research in historical settings and to provide benchmarks for future work.
  
To date, emotion detection TBED has demonstrated promising algorithmic performance, particularly for English-language data, mainly due to the availability of rich linguistic and lexical resources (Maks & Vossen, 2011; Valitutti et al., 2004). ED approaches have been successfully applied across a wide range of domains, including mental health analysis (Yang et al., 2023), customer behavior and marketing research (Wemmer et al., 2024), fake news detection (Zhang et al., 2023), and the humanities, where they support the analysis of emotional dimensions in literary texts (Santangelo, 2023). Within the fields of Digital Humanities and Computational Literary Studies, ED has attracted substantial attention (Kim & Klinger, 2019). It has been widely used to analyse emotions in historical plays (Yavuz, 2021; Schmidt et al., 2021), novels (Reagan et al., 2016), fairy tales (Mohammad, 2011), and political texts (Sprugnoli et al., 2016). Despite these advances, relatively few studies have examined the emotional dimension of other types of historical sources, such as historical correspondence (Gatti & Huesle, 2025; Turunen et al., 2022; Leemans et al., 2017; Mohammad, 2012). In particular, the existing literature lacks work assessing the broader applicability of emotion mining methods across specific linguistic and historical contexts, such as Spanish historical epistolary corpora.

The motivation for this shared task is to address the challenge that semantic shift (Hu, Amaral, & Kübler, 2022; Montanelli & Periti, 2023; Montes, Manrique-Gómez, & Manrique, 2024) poses for emotion detection in historical texts, given that the lexicons in this domain often exhibit substantial deviations between historical and modern affective meanings. This challenge underscores the need for robust methods that can capture and detect the complexity of affective states expressed in historical correspondence. Another motivation for this shared task is to investigate how large language models (LLMs) and language models (LMs) can contribute to TBED in the domain of historical correspondence, given their promising performance on similar tasks in social media contexts (Plaza del Arco et al., 2020).


