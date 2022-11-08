# Trojan-Model-Detection

### Problem Statement

Backdoor attacks have emerged as the main threat to the safe deployment of DNNs. An adversary poisons some of the training data in such attacks by installing a trigger. The goal is to make the trained DNN output the attacker's desired class whenever the trigger is activated while performing as usual for clean data. Various approaches have recently been proposed to detect/defend malicious backdoored DNNs. However, how to efficiently defend against this attack is still an open question. In this project, we would like you to focus on the project related to backdoor defense:

### Project Goals  

- **Trojan Model Detection:** Given an untrusted pre-trained network, the goal of trojan detection is to reveal the potential backdoored model (and to identify the infected label, i.e. backdoor related label). Neural cleanse (NC) [1] has proposed a novel and generalizable technique for detecting and reverse engineering hidden triggers embedded inside deep neural networks. Early work suggests that standard Trojan attacks may be easy to detect, but recently it has been shown that in simple cases one can design practically undetectable Trojans [2, 3]. So in this project, we invite you to help answer an important research question of how hard is it to detect hidden functionality that is trying to stay hidden? And we encourage you to design or explore a new trojan detection technique against the advanced backdoor attacks.


### Trojan Model Zoo

We have randomly sampled a subset of the backdoored model from the Trojan Detection Challenge [4]. You can download from here [link].

### Example of NC

For a quick start, we have provided an example of trojan model detection by NC as follows. 

### Reference 

[1] Wang B, Yao Y, Shan S, et al. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks[C]//2019 IEEE Symposium on Security and Privacy (SP). IEEE, 2019: 707-723.  
[2] Nguyen A, Tran A. WaNet--Imperceptible Warping-based Backdoor Attack[J]. ICLR, 2021.  
[3] Nguyen T A, Tran A. Input-aware dynamic backdoor attack[J]. NeurIPS, 2021.  
[4] Trojan Detection Challenge, https://trojandetection.ai/.
