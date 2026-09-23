# MUTAFormer web tool: original-script compatibility

This web directory depends on the unchanged Python scripts and historical archive in the project root. It is not a standalone replacement model implementation.

Defaults: 10 layers, 16 heads, embedding 64, dropout 0.1, vocabulary 250, sequence length 15, seed 11, 20 epochs, Adam learning rate 0.0001, training/evaluation batch 64, explanation batch 4. The web learning-rate default (including RFData and synthetic examples) was changed at the user's request from the saved paper configuration's 0.001. Historical verification results below used 0.001 and have not been rerun. Root scripts are unchanged. This is not the four-layer Small-panel evaluation model.
