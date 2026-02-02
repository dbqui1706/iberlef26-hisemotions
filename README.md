
<p align="center" style="overflow: hidden; height: 200px; width: 300px; position: relative;">
  <img align="center" src="hisemotions_2026.png" style="position: absolute; bottom: -50px; width: 80%;" />
</p>

<h1 align="center">HISEMOTIONS 2026: Historical Text Based Emotion Detection in Early Modern Spanish Correspondence</h1>

- [Description of the Task](#description-of-the-task)
  - [Overview](#overview)
  - [Relevance and Novelty](#relevance-and-novelty)
  - [Challenges Involved](#challenges-involved)
- [Dataset](#dataset)
- [References](#references)

## Description of the Task

### Overview
  HISEMOTIONS 2026 proposes a shared task focused on Text-Based Emotion Detection (TBED) in Early Modern Spanish epistolary texts (16th-17th centuries). The task aims to foster the development and evaluation of NLP methods specifically adapted to historical Spanish language varieties, addressing the semantic and diachronic challenges of emotion and affective states expression in Early Modern Spanish correspondence.  
  
While previous IberLEF shared tasks on sentiment and emotion analysis have focused on contemporary Spanish and modern text genres such as social media, reviews, and news, leaving the linguistic, orthographic, and semantic characteristics of Early Modern Spanish largely unexplored. Consequently, the transferability of models and resources developed for modern Spanish to historical language varieties remains unclear, particularly in domains where affective expression follows distinct semantic, cultural, and linguistic conventions. This shared task addresses this gap by evaluating emotion detection methods on Spanish historical correspondence. It seeks to answer core research questions concerning the generalisation of modern Spanish models to Early Modern Spanish, the impact of semantic change on emotion classification, and the effectiveness of domain- and diachrony-aware adaptation strategies compared to direct transfer. In doing so, the task aims to advance emotion detection research in historical settings and to provide benchmarks for future work.
  
To date, emotion detection TBED has demonstrated promising algorithmic performance, particularly for English-language data, mainly due to the availability of rich linguistic and lexical resources (Maks & Vossen, 2011; Valitutti et al., 2004). ED approaches have been successfully applied across a wide range of domains, including mental health analysis (Yang et al., 2023), customer behavior and marketing research (Wemmer et al., 2024), fake news detection (Zhang et al., 2023), and the humanities, where they support the analysis of emotional dimensions in literary texts (Santangelo, 2023). Within the fields of Digital Humanities and Computational Literary Studies, ED has attracted substantial attention (Kim & Klinger, 2019). It has been widely used to analyse emotions in historical plays (Yavuz, 2021; Schmidt et al., 2021), novels (Reagan et al., 2016), fairy tales (Mohammad, 2011), and political texts (Sprugnoli et al., 2016). Despite these advances, relatively few studies have examined the emotional dimension of other types of historical sources, such as historical correspondence (Gatti & Huesle, 2025; Turunen et al., 2022; Leemans et al., 2017; Mohammad, 2012). In particular, the existing literature lacks work assessing the broader applicability of emotion mining methods across specific linguistic and historical contexts, such as Spanish historical epistolary corpora.

The motivation for this shared task is to address the challenge that semantic shift (Hu, Amaral, & Kübler, 2022; Montanelli & Periti, 2023; Montes, Manrique-Gómez, & Manrique, 2024) poses for emotion detection in historical texts, given that the lexicons in this domain often exhibit substantial deviations between historical and modern affective meanings. This challenge underscores the need for robust methods that can capture and detect the complexity of affective states expressed in historical correspondence. Another motivation for this shared task is to investigate how large language models (LLMs) and language models (LMs) can contribute to TBED in the domain of historical correspondence, given their promising performance on similar tasks in social media contexts (Plaza del Arco et al., 2020).

### Relevance and Novelty

The TBED is a well-established task in modern NLP; however, historical texts remain largely underexplored, particularly in non-English contexts. The relevance of applying TBED to historical domains arises from a hypothesis at the intersection of historical studies and Digital Humanities: namely, that the extraction and analysis of emotions can support the validation of historical arguments and contribute to the reconstruction of individual identities from the past (Eustace et al., 2012; Ehrlicher et al., 2019; Ortega-Sánchez et al., 2020; Nanetti, Pavlopoulos & Cambria, 2023). This shared task aims to advance automatic TBED in historical contexts, thereby significantly enhancing Digital Humanities research by enabling new forms of quantitative and qualitative historical analysis.

This task aims to make the following contributions, which, to the best of our knowledge, are novel:

1. An empirical study that involves LLMs and LMs for the TBED task in the domain of the Early Modern Spanish correspondence corpus.  
2. The application and evaluation of diverse strategies and methods to adapt LLMs and LMs and improve their performance on this specific task.  
3. Bridging Natural Language Processing and Digital Humanities by engaging participants from diverse academic backgrounds and areas of expertise.  

### Challenges Involved

The studies cited above converge on several key challenges in historical texts related to TBED. These include: developing diachronic lexical representations; adapting computational methods for extracting information from historical language; and validating the extracted information using both human and automated approaches. Additionally, there is no established system for emotional annotation in historical epistolary corpora, nor a ground-truth standard for historical emotion lexicons and tags, since both the authors and their contemporaneous witnesses are long deceased, making it impossible to directly verify the true nature of the emotions expressed in the text. 

Each of these challenges is critical to the contributions outlined in the previous section.

## Dataset

The corpus of Spanish correspondence (16th–17th c.), digitised and transcribed, is sourced from work by Vaamonde (2015), P. S. Post Scriptum corpus[^1]. It includes private letters written in Portugal and Spain during the Early Modern period. The corpus consists mainly of previously unpublished correspondence from individuals of diverse social backgrounds, including men and women, adults and children, masters and servants, soldiers, artisans, clergy, and political actors. Characterised by an (almost) oral rhetoric and a focus on everyday concerns, these texts represent a register that has been largely understudied. Beyond assembling this unique collection, the P.S. Project provides the letters as a scholarly digital edition and an annotated corpus (PoS and syntactic dependencies), enabling systematic research on Early Modern epistolary practices. This shared task aims to use a selection of letters from the Spanish part of the corpus related to the 16th–17th centuries.  

The dataset will be divided into **training**, **development**, and **test** splits. The **training** set will be released with gold emotion labels and used to train models. The **development** (validation) set, also released with gold labels, will support model tuning and error analysis. The **test** set will be released without labels and used for final evaluation. Submissions will be evaluated against hidden gold labels, and results will be displayed on a public leaderboard. Participants will submit their predictions in a standardised format, with a limited number of submissions per team to prevent overfitting. The official evaluation scripts and metrics will be made publicly available prior to the evaluation phase to ensure transparency and reproducibility.  

Texts are segmented into emotion-bearing units (“**fragments**”), defined as contiguous spans of text corresponding to a clause or sentence that expresses a coherent affective state. **Fragments** derived from the same letter are kept together within a single split to prevent data leakage. Available metadata include letter-level information (when known) such as approximate date, place of origin, author identity, as provided by the Post Scriptum corpus.  

**Dataset Size (Tentative)**  

•	Training set: ~3000 annotated segments
•	Validation set: ~ 500 segments
•	Test set: ~1,000 segments


[^1]:http://teitok.clul.ul.pt/postscriptum/es/index.php?action=downloads

## References

Ehrlicher, H., Klinger, R., Lehmann, J., & Padó, S. (2019). Measuring historical emotions and their evolution: An interdisciplinary endeavour to investigate the “emotions of encounter.” *Liinc em Revista*, *15(1)*.

Ekman, P. & Friesen, W.V. (1978). Facial Action Coding System: A Technique for the Measurement of Facial Movement. *Consulting Psychologists Press*, Palo Alto, CA. https://doi.org/10.1037/t27734-000

Ekman, P. (1992). An argument for basic emotions. *Cognition & emotion*, *6(3-4)*, 169-200. http://dx.doi.org/10.1080/02699939208411068

Eustace, N., Lean, E., Livingston, J., Plamper, J., Reddy, W. M., & Rosenwein, B. H. (2012). AHR conversation: The historical study of emotions. *The American Historical Review*, *117(5)*, 1487–1531.

Feldman, D. B., & Jazaieri, H. (2024). Feeling hopeful: development and validation of the trait emotion hope scale. *Frontiers in psychology*, *15*, 1-18. https://doi.org/10.3389/fpsyg.2024.1322807

Gatti, F., & Huesler, J. (2025). *Text analysis methods for historical letters: The case of Michelangelo Buonarroti (EHES Working Paper No. 279)*. European Historical Economics Society.

Kim, E., & Klinger, R. (2019). *A survey on sentiment and emotion analysis for computational literary studies. Zeitschrift für digitale Geisteswissenschaften*. https://doi.org/10.17175/2019_008

Leemans, I. B., Maks, E., van der Zwaan, J. M., Kuijpers, H. M. E. P., & Steenbergh, K. (2017). Mining Embodied Emotions: A Comparative Analysis of Bodily Emotion Expressions in Dutch Theatre Texts 1600-1800. *Digital Humanities Quarterly*, *11(4)*. http://digitalhumanities.org:8081/dhq/vol/11/4/000343/000343.html

Maks, I., & Vossen, P. (2011). A verb lexicon model for deep sentiment analysis and opinion mining applications. In *Proceedings of the 2nd Workshop on Computational Approaches to Subjectivity and Sentiment Analysis (WASSA 2011)* (pp. 10–18).

Meta. (2024). Llama 3.1 8B Instruct [Large language model]. Hugging Face. https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct. 

Mohammad, S. (2011). From once upon a time to happily ever after: Tracking emotions in novels and fairy tales. In *Proceedings of the 5th ACL-HLT Workshop on Language Technology for Cultural Heritage, Social Sciences, and Humanities* (pp. 105–114). Portland, OR, USA: Association for Computational Linguistics.

Mohammad, S. M. (2012). From once upon a time to happily ever after: Tracking emotions in mail and books. *Decision Support Systems*, *53(4)*, 730–741.

Moßburger, L., Wende, F., Brinkmann, K., & Schmidt, T. (2020). Exploring online depression forums via text mining: A comparison of Reddit and a curated online forum. In *Proceedings of the Fifth Social Media Mining for Health Applications Workshop & Shared Task* (pp. 70–81). Barcelona, Spain (Online): Association for Computational Linguistics. https://www.aclweb.org/anthology/2020.smm4h-1.11

Muhammad, S. H., Ousidhoum, N., Abdulmumin, I., Wahle, J. P., Ruas, T., Beloucif, M., ... & Mohammad, S. (2025b). Brighter: Bridging the gap in human-annotated textual emotion recognition datasets for 28 languages. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics* (Volume 1: Long Papers) (pp. 8895–8916).

Muhammad, S. H., Ousidhoum, N., Abdulmumin, I., Yimam, S. M., Wahle, J. P., Ruas, T. L., ... & Mohammad, S. (2025a). SemEval-2025 task 11: Bridging the gap in text-based emotion detection. In *Proceedings of the 19th international workshop on semantic evaluation (SemEval-2025)* (pp. 2558–2569).

Nanetti, A., Pavlopoulos, J., & Cambria, E. (2023). Sentiment Analysis of Primary Historical Sources. *2023 IEEE International Conference on Data Mining Workshops (ICDMW)*, 767–772.  https://doi.org/10.1109/ICDMW60847.2023.00104 

Ortega-Sánchez, D., Pagès Blanch, J., & Pérez-González, C. (2020). Emotions and construction of national identities in historical education. *Education Sciences*, *10(11)*, 322.

Plaza del Arco F.M., Strapparava C., Ureña López A.L., & Martín, M. (2020). EmoEvent: A Multilingual Emotion Corpus based on different Events. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, (pp. 1492–1498), Marseille, France. European Language Resources Association.

Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., & Dodds, P. S. (2016). The emotional arcs of stories are dominated by six basic shapes. *EPJ Data Science*, *5(1)*, 31. https://doi.org/10.1140/epjds/s13688-016-0093-1

Santangelo, P. (2023). The role and symbolic meanings of emotion in literary language. In G. L. Schiewer, J. Altarriba, & B. C. Ng (Eds.), *Language and Emotion, Volume 3* (pp. 1558–1590). Berlin, Boston: De Gruyter Mouton. https://doi.org/10.1515/9783110795486-010

Schmidt, T., Dennerlein, K., & Wolff, C. (2021). Towards a corpus of historical German plays with emotion annotations. In *3rd Conference on Language, Data and Knowledge* (LDK 2021) (Open Access Series in Informatics, Vol. 93, pp. 9:1–9:11). Schloss Dagstuhl – Leibniz-Zentrum für Informatik. https://doi.org/10.4230/OASIcs.LDK.2021.9

Sprugnoli, R., Tonelli, S., Marchetti, A., & Moretti, G. (2016). Towards sentiment analysis for historical texts. *Digit. Scholarsh. Humanit.*, *31*, 762–772. https://doi.org/10.1093/llc/fqv027 

Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., ... & Iqbal, S. (2025). Gemma 3 technical report. *arXiv preprint arXiv:2503.19786*. https://doi.org/10.48550/arXiv.2503.19786 

Turunen, R., Taskinen, I., Uusitalo, L., & Kivimäki, V. (2022). Mining Emotions from the Finnish War Letter Collection, 1939–1944. In K. Berglund, M. La Mela, & I. Zwart (Eds.), *Proceedings of the 6th Digital Humanities in the Nordic and Baltic Countries Conference* (DHNB 2022): Uppsala, Sweden, March 15–18, 2022 (pp. 135-144). (CEUR Workshop Proceedings; Vol. 3232). CEUR-WS. http://ceur-ws.org/Vol-3232/paper10.pdf 

Vaamonde, G. (2015). P. S. Post Scriptum. Dos corpus diacrónicos de escritura cotidiana. *Procesamiento del Lenguaje Natural*, *55*, 57–64. [ISSN: 1135–5948].

Valitutti, A., Strapparava, C., & Stock, O. (2004). Developing affective lexical resources. *PsychNology Journal*, *2(1)*, 61–83.

Wemmer, E., Labat, S., & Klinger, R. (2024). EmoProgress: Cumulated emotion progression analysis in dreams and customer service dialogues. In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation* (LREC-COLING 2024) (pp. 5660–5677).

Yang, K., Ji, S., Zhang, T., Xie, Q., Kuang, Z., & Ananiadou, S. (2023). Towards interpretable mental health analysis with large language models. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (EMNLP 2023) (pp.6056-6077). https://doi.org/10.18653/v1/2023.emnlp-main.370 

Yavuz, M. C. (2021). Analyses of character emotions in dramatic works by using EmoLex unigrams. In *Proceedings of the Seventh Italian Conference on Computational Linguistics* (CLiC-it 2020) (pp.472–477). Bologna, Italy: Associazione Italiana di Linguistica Computazionale. https://aclanthology.org/2020.clicit-1.73/

Zhang, Y., Su, X., Wu, J., Yang, J., Fan, H., & Zheng, X. (2023). EmoKnow: emotion- and knowledge-oriented model for COVID-19 fake news detection. In X. Yang, H. Suhartanto, G. Wang, B. Wang, J. Jiang, B. Li, H. Zhu, & N. Cui (Eds.), *Advanced Data Mining and Applications: 19th International Conference*, ADMA 2023, Shenyang, China, August 21–23, 2023, proceedings, part I (pp. 352-367). (Lecture Notes in Computer Science; Vol. 14176). Springer, Springer Nature. https://doi.org/10.1007/978-3-031-46661-8_24

