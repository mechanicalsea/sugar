# AM-Softmax

> Wang, F., Cheng, J., Liu, W., Liu, H., 2018. Additive Margin Softmax for Face Verification. IEEE Signal Process. Lett. 25, 926–930. https://doi.org/10.1109/LSP.2018.2822810

- Margin is a penalty term to loss function, where inceasing margin results in larger loss because it forces the objective to learning more margin.
- Scale is, in my opinion, like a scale of the margin,  which means the large scale brings more space within intra-class.

Intuitively, when classified correctly, larger scale lower loss, but larger margin higher loss. On the other hand, when classified uncorrectly, increasing both of them leads to higher loss.