# Supervisor Meeting Script (Banglish, casual)

**Date:** 2026-11-05
**Ki korlam:** Project er full experiment shesh, ekhon results + validation dekhachhi.
**Ei file use korbo:** Meeting er somoy bolte. Number gulo chokh kore bolbo.

---

## 1. Shuru - plan koto

Sir, amader ekta research question chilo:

> Bengali instruction tuning er jonno 100K row er data theke sirf 10K row select kore
> train korle ki model better hobe? Random selection er cheye, ar full pool (91K) er cheye
> koto door?

Amra 6ta arm compare korechi:

| Arm | Ki ki |
|---|---|
| **A** | Full pool (91,120 row) - eta upper bound, amader reference |
| **B** | Random 10K - eta amader baseline |
| **C** | DEITA style: complexity x quality score, ar diversity filter |
| **D** | Quality only (judge er quality score er upor top 10K) |
| **E** | Complexity x quality, but diversity chara (C er shudhu score version) |
| **F** | Cherry-LLM / IFD (instruction fine-tuning difficulty theke top 10K) |

Base model: **Qwen2.5-0.5B-Instruct**, LoRA train korechi. Sathe shesh: **Stage 4, 5, 6, 7, 8, 9 shesh**, mane experiment puro hoye gese.

---

## 2. Data kivabe banano

Source: Bangla-Instruct (342,390 row).

Step by step:
1. **100K sample** nilam, tarpor 500 row alada kore rakhlam (held-out) - ei row kono training e nai, shudhu test er jonno.
2. **Filter** - lamba, kharab mojibake, English chara, truncated, repetition egulo bondho kore felechi. Total **5,081 row (5.1%)** drop hoyeche. Filter er por **91,120 row** bache.
3. **Dedup** - exact 2,991 ar near-duplicate (MinHash) remove.
4. **Decontamination** - benchmark (MMLU, BanglaRQA, mHumanEval-BN) er sathe 13-gram match kore row delete, jate answer leak na hoy. 71 row remove hoyeche.
5. **Score** - LLM judge (Qwen2.5-7B) diye protita row er **complexity (c)** ar **quality (q)** score. Eta 3ta Kaggle session e lagse, ~21 ghonta.
6. **Selection** - DEITA style greedy: score desc order e sort kore ekta ekta add kori, jodi oi row already selected row er sathe similarity < tau (0.74) thake.

**Ekta jinis bole rakhi:** filter e **code** er upor beshi pressure poreche (20.7% drop, mostly English filter), tai arm C te code shudhu 3%. Eta amader limitation - niche bujhiyechi.

---

## 3. Result - main table

**Ei table ektai jinish bujhiye dey:**

| Arm | Held-out loss | MMLU | RQA | Pairwise win vs B |
|---|---|---|---|---|
| base | 0.9779 | 0.2583 | 0.2071 | 0.4729 |
| A (91K) | 0.7169 | 0.2624 | 0.1827 | 0.7032 |
| **B (random 10K)** | 0.8126 | 0.2584 | 0.1504 | 0.5000 |
| C (DEITA + div) | 0.8380 | 0.2505 | 0.1757 | 0.3761 |
| D (quality only) | 0.8572 | 0.2666 | 0.1597 | 0.4638 |
| E (no div) | 0.8661 | 0.2370 | 0.1681 | 0.3484 |
| F (IFD) | 0.8260 | 0.2358 | 0.1531 | 0.3207 |

**Low loss = better. High win rate = better.**

### Shuru te jeta chokh korte hobe (IMPORTANT)

**3ta key finding ache, kintu shokhon amader result "shundor na" - seta openly bolte hobe.**

**(a) Full pool sob cheye strong**
Arm A (91K) sob arm er cheye better - sheta expected, karon 9x beshi data. Paper er main message: "10K select kore full pool er kache pauchha jay" - ha, kintu ekta C-arm dewa jay na.

**(b) Complexity x quality selection, random er cheye WORLD hoye gese**
- Held-out loss: C = 0.8380, B = 0.8126 → **C er loss beshi (kharap)**
- Pairwise: C er B er against win rate 0.3761 → **C lose korche B er against**

Eta amader main negative result. Ar keno holo? Eta amar jonno surprise chilo. Section 4 e bujhiyechi.

**(c) E theke C e jawna (diversity) kaj koreche**
- C (with diversity) vs E (without): held-out loss 0.8380 vs 0.8661 → C better
- Head-to-head pairwise: C beats E 0.6170
- Diversity → C ar E er overlap 58% (na hole 91.5% chilo, mane diversity filter ta kaj korchilo na)

**Positive signal kothay?** Token efficiency te:

| Arm | Tokens | Loss reduction per M token |
|---|---|---|
| A | 111.5M | 0.0023 |
| B | 12.2M | 0.0136 |
| F | 10.9M | 0.0139 |
| **C** | **15.5M** | **0.0090** |
| D | 15.5M | 0.0078 |
| E | 15.2M | 0.0074 |

**Eta puro ektu jinish bujhiye dey, tai ektu ektu kore bolbo:**
- F ar B er per-token number beshi - karon tara kom token e train koreche, absolute loss emon-i bhalo na.
- **C, D, E er moddhe C sob cheye token-efficient.** Eta prove kore je diversity step ta ektai kaj korche, c x q score na.

---

## 4. Eta keno holo? - Validation (ei part ta important)

Supervisor er kache janiye debo: **amader judge ta kivabe behave korche seta human label er sathe compare korechi.**

100 row ke amra manually (human) label korechilam - **complexity** ar **quality** alada alada, 1 theke 5 scale e.

Phir sei 100 row er judge score er sathe compare korlam.

Scale niye ektu jinis: human 1-5 use korche, judge 1-6 (proba dekhe expectation nite). Tai judge ke 1-5 te map korechi (`1 + (judge-1) * 4/5`) ar round korechi.

### Result:

| Ki compare korlam | Spearman rho | p-value | Exact same | ±1 moddhe |
|---|---|---|---|---|
| **Complexity**: judge vs human | **0.543** | <0.0001 | 40% | 95% |
| **Quality**: judge vs human | **0.078** | 0.44 | 14% | 54% |

### Ki mane eta?

- **Complexity**: judge ar human ek jinish measure korche. Eta valo. Spearman 0.54 mane "moderate agreement" - perfect na, kintu usable.
- **Quality**: judge ar human **kono jinish ekshathe measure-i korche na**. Rho 0.078, p-value 0.44 mane statistically significance-i nai (shuru torkhono bhabe just random).

**Ar ekta jinis:** Judge er quality score e ekta ceiling ache. 100 row er moddhe 70 ta row 4-e, 23 ta 5-e, 7 ta 3-e, ar **ekta-i 3 er niche nai**. Human label e gele 1 theke 5 porjonto spread ache.

### Eta ki judge er bhul?

Na. Ami test korechi.

**Duita alada model diye check korchi:**
1. Qwen2.5-7B (amader original judge)
2. Llama-3.1-8B (notun, alada family)

| | Qwen judge | Llama rater |
|---|---|---|
| Complexity vs human | 0.543 | **0.525** |
| Quality vs human | 0.078 | **-0.084** |

Duita model e quality te fail korche, complexity te duitai thik. Eta mane **model er bhul na - rubric/problem e gap ache.**

**Ekta jinis bole rakhi, karon eta important:** Amader rubric e quality er first point-i "correctness and factual accuracy" likha. Eta missing instruction na. Model simply seta **enforce korte parle na.** Judge er score gulo practically dekhbe **ki bhasha shundor, ki format thik, instruction follow korche** - kintu **ki information thik ache** seta verify korte parbe na.

### Ki bujhiyechhi exactly?

Manush (human) jokhon quality 1 dilo, oi row gulo mostly ei type er:
- **id 55** - "Great Albatross" ke Palearctic area r bipannna pradhan species boleche (eta to Southern Ocean er pakshi)
- **id 83** - generic "film criticism" paragraph - actually question er answer-i dey nai
- **id 20** - মঞ্চু সমাজ ke Bangladesh e boleche (eta Jharkhand, India)
- **id 4** - John Simm er role er jonye "Dr. David Thompson" naam banay, Colin Pitchfork ke "Colin Richards" rename kore

Oi row gulo human "quality 1" diye uthse, kintu Qwen judge **5.0-5.3** dey, ar Llama rater o **4.1-4.4** dey.

Eta-i amader core insight: **judge fluent, well-formatted Bangla ke valo bolche, kintu content thik ache na - seta check korte parche na.**

**Implication:** Paper e amader `q` score ke "ranking signal" hisabe likhte hobe - eta human quality er sathe match kore na. Ar limitation section e likhte hobe je judge c te reliable, q te unreliable.

---

## 5. Ar ekta important result - downstream benchmark (eta surprise korbe)

Sir, ektu jinis alada kore bolte chai, karon eta table e dike chokh korle mone hobe "D to valo":

**Arm D er MMLU accuracy sob model er cheye beshi** - 0.2666 (B er cheye +0.0082, p=0.999).

**Kintu ami nije test korechi - eta asole "valo" na, ekta "nominally valo":**
- 14,042 question e paired bootstrap korechi. D vs base: +0.0083, CI [-0.0009, +0.0174], p=0.964 → **significant na**.
- D vs arm A: +0.0042, CI [-0.0038, +0.0121], p=0.849 → **arm A keo D eseche na**.
- Ar D er 3ta seed er MMLU range (0.2601-0.2706) B er range er sathe **overlap kore** (B: 0.2522-0.2639).

Eta mane D er MMLU "best" claim ta **asole prove hoye nai**. Paper e eta "highest observed, not statistically significant" hisabe likhbo.

**Ar amader judge quality te "fail" kore.** D to select-i hoyeche quality score er upor. Eta tension - MMLU e D valo dekhay, kintu quality signal ta human verified na. Paper e dutai report korbo, ar limitation e bole debo "D er MMLU result ta judge er quality signal ke depend kore, human-verified quality claim na."

**Ar ekta jinis:** BanglaRQA te **base model (train korar age) sob cheye beshi** - 0.2071 (answerable question e 0.2744). Protita fine-tuned arm er cheye beshi. Matlab native Bengali reading comprehension e, ei synthetic data diye train korle accuracy komche. Eta o report korte hobe.

---

## 5b. Ektu jinis - "sob kharap ekta domain e" (eta important)

Sir, amader C arm vs B arm er gap (held-out loss +0.0254) - ami decompose korlam **domain onujayi**. Result:

| Domain | Row count | Loss B -> C | Gap e konta |
|---|---|---|---|
| **code** | 70 | 0.8614 -> 0.9950 (+0.1336) | **78.4%** |
| math | 174 | 0.6036 -> 0.6150 | 16.6% |
| open_qa | 163 | 0.9542 -> 0.9564 | 3.0% |
| baki shob | 63 | - | ~2% |

**Matlab: code baddei 78% of the gap.** Baki 400 row e gap matro **+0.0055** - mane basically zero.

Eta **Arm F er cheye** prove korar ekta bishoy: F er code share 21.2%, ar F er code loss 0.8423 - B er 0.8614 er **cheye better**! Pairwise teo F code prompt e B ke beat kore (0.4643 vs C er 0.3143).

**Eta "c x quality selection kaaj kore na" finding er ekta mechanical explanation:** c x q code systematically kom select kore (C te 3%, E te 1.4%), tai loss penalty ta mostly code e bhiteche.

**Paper er impact:** amader negative result ta "quality selection kaaj kore na" na - beshi accurate hobe "quality selection code er representation haray, ar sei hare ta-i mostly ekta domain er problem."

---

## 5c. Seed ar truncation - duita jinis jeta table e dekha jay na

**Duita jinis ami eshe add korechi, karon eta na bujhe supervisor bolte paren "eta to noise":**

**(a) Downstream benchmark e seed variation, arm gap er cheye boro**

Held-out loss e seed spread matro 0.001 - kintu MMLU/RQA te:
- MMLU seed range: 0.004 - 0.012
- RQA seed range: 0.012 - 0.029

Mane arm gap er sathe comparable. **Konta claim survive kore:**
- **C er RQA advantage B er cheye** - ROBUST (C er sob seed B er sob seed er upor)
- **D er MMLU advantage** - ROBUST na (seed range overlap kore, ar paired CI zero cross kore)
- **E ar F er RQA** - entangled, RQA te order ta safe na (held-out loss ar pairwise e safe)

**(b) Generation "atke" thaka - eta loss number chhuye**

`hit_limit` mane koto percentage generation 1,536 token limit e atke geche, stop hoye nai:

| Arm | Mean token | Hit limit | Bangle char share |
|---|---|---|---|
| A | 1,108 | 0.587 | 0.626 |
| B | 1,172 | 0.689 | 0.662 |
| base | 1,343 | 0.800 | 0.709 |
| F | 1,287 | 0.804 | 0.658 |
| **C** | 1,384 | **0.857** | 0.642 |

**C sobcheye kharap** - 85.7% generation atke geche. Qualitative sample e exact ki dekhay:
- code prompt e C ekta Bengali "সংক্ষেপ:" likhe instruction phrase **bartobar** repete kore
- list_brainstorm prompt e C "কেন কেন? কেন কেন?" repete kore token limit porjonto

**Eta pairwise judging er ekta problem:** judge duita truncated/repetitive generation compare korte ese, jeta "beshi complete dekhay" setai reward kore. **Pairwise win rate report korar shathe hit_limit o report korte hobe**, noyto pairwise number protita arm ke beshi valo dekhay.

---

## 6. Blind spot-check (etao important)

**Problem ki chilo:** Stage 4 e amra 100 row er ekta sheet banachilam - 50 ta arm C er, 50 ta arm B er - **arm name chhupa** (blind). Goal chilo: manually rate kore dekhna C er row ki better na B er row.

**Issue:** Judge diye rate korle **circular** hobe. Karon arm C to select-i kora hoyeche "judge er c x q maximum koro" diye. Ekhon judge diye check korle ami nijei nijer selection criterion ke reward korbo. Ei bhul result amake dewa hobe na.

**Solution:** Ekta **independent model** (Llama-3.1-8B, Qwen er cheye alada family) diye rate korlam. Eta 1-5 "overall usefulness" scale e, human sheet er scale er sathe same.

**Result:**

| Arm | Independent rating (mean) | 95% CI |
|---|---|---|
| B (random) | 4.055 | [4.005, 4.101] |
| **C (DEITA + div)** | **4.124** | [4.088, 4.164] |

**Difference: +0.070** (95% CI [+0.009, +0.133]), p(C better) = 0.99

**Interpretation:** C er row gulo B er cheye **onek kom na, kintu direction e better** - statistically significant, kintu effect choto. Arm C "onek valo" na, "ekdom worse o na".

**Ektu jinis amake bhalo lage:** Circular Qwen view te gap onek boro dekhay (c x q 23.25 vs 17.60) - mane ami nijer judge ke call korle 10x beshi gap dekhtam. Independent model e gap 0.070. Paper e **duita report korbo, independent ta ke headline korbo.**

**Ektu jinis amake bhalo lage:** Ekta **human** spot-check o korar plan achhe (a human rate korbe same sheet e). Eta optional, kintu korle claim aro strong hobe. Independent model result already ache.

---

## 7. Ar koto kaaj baki?

**Experiment puro shesh.** Ekhon shudhu **writing** baki:

1. Paper outline + venue + page limit decide kora
2. Introduction + related work (DEITA, Cherry-LLM/IFD, TigerLLM, Bangla-Instruct)
3. Data & pipeline section
4. Selection methods section (arm A-F, tau calibration)
5. Experimental setup
6. Results & analysis ← **ei jaygay amader validation result gulo jabe**
7. Limitations ← **judge quality gap ta jabe**
8. Abstract, conclusion, final pass

**Timeline:** Amar plan - 2-3 soptahe outline + first draft, tarpor feedback nia improve.

**Extra kaaj jodi korte chai:**
- Human spot-check pass (optional, 1-2 ghonta)
- Judge quality rubric nira kaj (correctness ar fluency alada kora) - future work hisabe

---

## 8. Questions supervisor er kase theke ase (anticipated)

**Q: "Tumi to bolcho C arm better na, tahole paper er message ki?"**
> Amader message hobe: "Quality-based selection, jemon DEITA/Cherry-LLM, amar experiment e random selection er cheye better hoy na - and human validation dekhay je keno (rubric quality signal verify korte parche na). But diversity step (tau) ekta real, measurable improvement dey." Eta negative kintu clear result - literature e onek "quality selection works" claim ache, amader data tar against evidence. Eta publication worthy.

**Q: "Quality score to kaaj korchilo na, tahole D arm er MMLU result ki matlab?"**
> D er MMLU **highest observed** (0.2666), kintu paired bootstrap e D vs base p=0.964, D vs A p=0.849, ar D er seed range B er range er sathe overlap kore. Eta mane D er MMLU claim ta statistically **significant na**. Paper e amra eta "highest observed, not significant" hisabe likhbo, limitation e bole debo je D er MMLU result ta judge er quality signal ke depend kore - human-verified quality claim na.

**Q: "Tumi to bolcho gap ta code er, tahole seta to filter er artifact, selection er problem na"**
> Thik bolchen - ei jinis ta amader **main limitation**, tai bolbo na. Kintu amader argument eta na je "selection kaj kore" - eta je "quality-based selection er effect mostly ekta domain (code) er representation issue, baki 400 row e gap ~zero". Eta paper er honest framing, ar eta literature er "quality selection works" claim er against aro strong evidence (karon quality score actually ki dekhay seta human validation e fail koreche).

**Q: "Generation eto atke geche ki? Eta ki amader result invalidate kore?"**
> Eta held-out loss ke invalidate kore na - loss per-token compute hoy, generation truncate howa sei jinis na. Kintu **pairwise judging** e problem, karon judge duita truncated output compare korche. Eta amader limitation section e explicit thakbe, ar paper e hit_limit table o thakbe. Ami nijei bolbo - noyto supervisor palt kore.

**Q: "Ekta model diye ki enough? Multi-seed ki korcho?"**
> Ha, 3 seed per arm (A excepted). Held-out loss e seed spread ~0.001, kintu MMLU/RQA te 0.004-0.029 - mane downstream comparison gulo seed range ekar sathe report korbo.

**Q: "Compute koto lagse?"**
> ~60-90 GPU ghonta total Kaggle e, 2-3 week quota te. Stage 9 (ei validation) matro 21 minute nichi.

**Q: "Ki kore judge niye confidence thik?"**
> Stage 3a te judge backend benchmark korchi - Spearman 1.000 pilot er sathe. Eta judge er *stability* prove kore, human agreement prove kore na. Eta duita alada jinis - ekhon amra dutai dekhe felechi.

**Q: "Token count ta ki consistent?"**
> Ekta jinis bole rakhi - Stage 1 e token count Llama tokenizer diye hoyeche (111.5M), training e Qwen tokenizer diye (102.4M), ~9% difference. Paper e protita table e kon tokenizer use korchi seta mention korbo. Order ta kichu change hoy na.

---

## 9. Shesh - ki chai supervisor theke?

1. **Target venue** - kon conference/journal e patha korbo? Page limit koto? (Affects likha style)
2. **Negative result e ki ok?** - Amar plan ache eta openly likhi, "quality selection kaaj kore na" claim er against evidence hisabe.
3. **Human spot-check korbo?** - Tahole 1-2 ghonta lagbe, arm C er claim aro strong hobe.
4. **Aro kono experiment chai?** - E.g. quality rubric fix kore abar run? (Eta 2-3 din GPU)

---

*Numbers sob `results/full_run/` folder e ache. Script ar code `analysis/` ar `kaggle/full_run/` e.*
