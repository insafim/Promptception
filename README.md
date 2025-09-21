# <img src="Assets/Promptception-Logo.png" height="40"> Promptception: How Sensitive Are Large Multimodal Models to Prompts? [EMNLP 2025 🔥]

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Image">
</p>

> #### [Mohamed Insaf Ismithdeen](https://scholar.google.com/citations?user=--fYSbUAAAAJ&hl=en), [Muhammad Uzair Khattak](https://scholar.google.com/citations?user=M6fFL4gAAAAJ&hl=en), [Salman Khan](https://salman-h-khan.github.io/)

#### Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), Swiss Federal Institute of Technology Lausanne (EPFL), Australian National University

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://insafim.github.io/Promptception/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://www.arxiv.org/abs/2509.03986)

Official GitHub repository for  `Promptception: How Sensitive Are Large Multimodal Models to Prompts?`.

## 📢 Latest Updates

- **Aug-2025:** Promptception is accepted at EMNLP 2025 (Findings)! 🎊🎊  
- **Nov-2025:** Mohamed Insaf Ismithdeen will be presenting *Promptception* as a **poster** at EMNLP 2025 (Findings Session 3, Nov 7). 📝✨ <img src="Assets/emnlp_2025_logo_v1.png" height="25">
---

## <img src="Assets/Promptception-Logo.png" height="25">  Overview

<p align="center">
  <img src="Assets/sunburst.png" width="70%" alt="Overview">
</p>

Despite the success of Large Multimodal Models (LMMs) in recent years, prompt design for  LMMs in Multiple‑Choice Question Answering (MCQA) remains poorly understood. We show that even minor variations in prompt phrasing and structure can lead to accuracy deviations of up to 15\% for certain prompts and models. This variability poses a challenge for transparent and fair LMM evaluation, as models often report their best-case performance using carefully selected prompts. To address this, we introduce **Promptception**, a systematic framework for evaluating prompt sensitivity in LMMs. It consists of 61 prompt types, spanning 15 categories and 6 supercategories, each targeting specific aspects of prompt formulation, and is used to evaluate 10 LMMs ranging from lightweight open‑source models to GPT-4o and Gemini 1.5 Pro, across 3 MCQA benchmarks: MMStar, MMMU‑Pro, MVBench. Our findings reveal that proprietary models exhibit greater sensitivity to prompt phrasing, reflecting tighter alignment with instruction semantics, while open‑source models are steadier but struggle with nuanced and complex phrasing. Based on this analysis, we propose Prompting Principles tailored to proprietary and open-source LMMs, enabling more robust and fair model evaluation.

---
## 🏆 Highlights
1. **Comprehensive Prompt Sensitivity Analysis:** We present the most extensive study to date on the impact of prompt variations across diverse multimodal benchmarks and LMM architectures. To facilitate this study, we introduce Promptception, a systematic evaluation framework comprising of 61 prompt types, organized into 15 categories and 6 supercategories, each designed to probe specific aspects of prompt formulation in LMMs.
2. **Evaluation Across Models, Modalities, and Benchmarks:** We assess prompt sensitivity across a diverse set of model sizes and architectures, including both open-source and proprietary LMMs. Our analysis spans multiple modalities and benchmarks; MMStar (single image), MMMU-Pro (multi-image), and MVBench (video) and we further evaluate sensitivity across various question dimensions within these benchmarks to ensure a comprehensive understanding.
3. **Best Practices for Prompting:** We identify key trends in prompting and propose Prompting Principles for effective and consistent evaluation of LMMs.
4. 

## Citation 📜

```bibtex
@misc{ismithdeen2025promptceptionsensitivelargemultimodal,
      title={Promptception: How Sensitive Are Large Multimodal Models to Prompts?}, 
      author={Mohamed Insaf Ismithdeen and Muhammad Uzair Khattak and Salman Khan},
      year={2025},
      eprint={2509.03986},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.03986}, 
}
```

## License :scroll:
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.


Looking forward to your feedback, contributions, and stars! :star2:
Please raise any issues or questions [here](https://github.com/insafim/Promptception/issues). 
