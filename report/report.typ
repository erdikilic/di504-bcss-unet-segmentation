#set document(title: "Semantic Segmentation of Breast Cancer Tissue Regions with U-Net: Encoder and Transfer Learning Study")
#import "@preview/neural-netz:0.3.0": draw-network

#set page(
  paper: "us-letter",
  margin: (top: 0.75in, bottom: 1in, x: 0.625in),
  numbering: "1",
  columns: 2,
)
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 1em)
#set heading(numbering: (..n) => {
  let nums = n.pos()
  if nums.len() == 1 { numbering("I.", nums.last()) }
  else { numbering("A.", nums.last()) }
})
#show heading.where(level: 1): it => {
  set align(center)
  set text(size: 10pt, weight: "bold")
  block(above: 1.1em, below: 0.6em)[#smallcaps(it)]
}
#show heading.where(level: 2): it => {
  set text(size: 10pt, weight: "regular", style: "italic")
  block(above: 0.9em, below: 0.4em)[#it]
}
// figure and table captions smaller than the 10pt body text (IEEE style)
#show figure.caption: set text(size: 8pt)

// helper for placing a full-width float across both columns
#let fullwidth(body) = place(top + center, scope: "parent", float: true, body)

// ---------- Title and abstract (span both columns) ----------
#fullwidth[
  #set align(center)
  #text(size: 18pt, weight: "bold")[
    Semantic Segmentation of Breast Cancer Tissue\ Regions with U-Net: A Study of Encoder Depth\ and Transfer Learning
  ]
  #v(0.6em)
  #text(size: 11pt)[Erdi Kılıç]
  #v(0.1em)
  #text(size: 10pt)[Department of Health Informatics, Graduate School of Informatics\ Middle East Technical University]
  #v(0.1em)
  #text(size: 10pt, style: "italic")[DI 504: Foundations of Deep Learning, Final Project Report]
  #v(0.9em)

  #block(width: 88%)[
    #set align(left)
    #set par(justify: true, first-line-indent: 0em)
    #text(size: 9pt)[
      *Abstract*---Manual delineation of tissue regions in breast cancer
      histopathology is slow and varies between observers, which motivates
      automated pixel-level segmentation. In this work we train and compare three
      U-Net variants for five-class tissue segmentation on the Breast Cancer
      Semantic Segmentation (BCSS) dataset: a Vanilla U-Net trained from scratch, a
      U-Net with a ResNet-34 encoder initialized randomly, and the same network
      initialized with ImageNet weights. The two-step comparison isolates the
      contribution of a stronger encoder from the contribution of transfer
      learning. All models share the same decoder design, loss, augmentation
      pipeline, and a patient-level data split that prevents leakage between
      training and test patches. Hyperparameters were selected with Optuna. On a held-out test
      set the Vanilla U-Net reaches a mean Dice of 0.760, the random ResNet-34
      encoder reaches 0.809, and the pretrained encoder reaches 0.824, all at or
      above the fully convolutional network (FCN) baseline of roughly 0.75
      reported on the same dataset. The largest improvement appears on necrosis,
      the rarest class, where Dice rises from 0.59 to 0.83. The results show that
      a deeper encoder accounts for most of the gain while ImageNet
      initialization adds a smaller but consistent boost.
    ]
    #v(0.4em)
    #text(size: 9pt)[
      *Index Terms*---semantic segmentation, U-Net, breast cancer histopathology,
      transfer learning, class imbalance, BCSS.
    ]
  ]
  #v(0.6em)
]

= Introduction

Breast cancer is the most frequently diagnosed cancer worldwide, and the
microscopic examination of stained tissue remains the reference standard for
diagnosis and treatment planning. Pathologists study hematoxylin and eosin
(H&E) stained whole-slide images and mark regions such as tumor, stroma,
inflammatory infiltrate, and necrosis. The relative amount and arrangement of
these regions carry prognostic value, but marking them by hand is slow, and the
same slide can be annotated differently by two experts. Because the number of
slides far exceeds the number of available pathologists, the manual workflow
does not scale.

Automatic semantic segmentation offers a way to produce consistent tissue maps
quickly. The task is to assign a tissue class to every pixel of an input patch,
which is a dense prediction problem rather than a single image-level label. The
data in this project comes from triple-negative breast cancer cases, an
aggressive subtype with few targeted treatment options, so reliable tissue
characterization is particularly useful here.

We study this problem with U-Net @ronneberger2015, the encoder-decoder network
with skip connections that has become the default starting point for medical
image segmentation. The central question we ask is a practical one: when the
Vanilla U-Net is replaced by a U-Net with a deeper ResNet encoder, how much of any
improvement comes from the encoder architecture itself, and how much comes from
initializing that encoder with ImageNet weights? To answer this we train three
variants under identical conditions and compare them on per-class and overall
metrics. The Vanilla U-Net trained from scratch isolates the baseline behavior, a
ResNet-34 encoder with random weights adds the architectural change, and the
same encoder with ImageNet weights adds transfer learning on top. Comparing the
first two variants measures the effect of the encoder, and comparing the last
two measures the effect of pretraining.

The remainder of the report covers related work (Section II), the dataset and its
class imbalance (Section III), the method (Section IV), the experimental setup
(Section V), results and discussion (Section VI), and conclusions (Section VII).

= Related Work

*U-Net (Ronneberger et al., 2015).* Ronneberger et al. @ronneberger2015
introduced U-Net for biomedical image segmentation. The network contracts the
input through an encoder of convolution and pooling layers, then expands it
through a decoder of upsampling layers, while skip connections copy encoder
features to the matching decoder stage so that spatial detail lost during
downsampling is recovered. Their main practical contribution was showing that
heavy data augmentation, including elastic deformations, lets the network train
well from only a few annotated images. On the ISBI cell-tracking challenge the
model reached a mean intersection-over-union (IoU) of 0.92 and outperformed the
earlier sliding-window approaches. The authors trained with stochastic gradient
descent and used a weighted cross-entropy loss to deal with class imbalance.
Their public implementation is the architectural basis for our Vanilla U-Net.

*BCSS (Amgad et al., 2019).* Amgad et al. @amgad2019 released the dataset used
in this project and showed that structured crowdsourcing can produce
high-quality annotations for training segmentation models. As preprocessing they
applied color normalization to reduce staining differences between
institutions, then trained fully convolutional networks (FCN) for the five
tissue classes. The study reported per-class Dice coefficients in roughly the
0.67 to 0.85 range and a mean area-under-curve around 0.945, which we treat as
the direct baseline because it is measured on the same data. They also observed
that tumor and stroma are easier to segment than inflammatory and necrotic
regions, which are smaller and more variable in appearance. The dataset and code
are available on GitHub, making the benchmark reproducible.

*Improved U-Net (Drioua et al., 2023).* Drioua et al. @drioua2023 proposed
modifications to the U-Net encoder for breast cancer histopathology
segmentation on a related H&E dataset. Their pipeline used patch extraction and
rotation-based augmentation, and the modified network reached an IoU of 0.861,
an F1 score of 0.909, and accuracy of 0.986, clearly above the FCN baseline
(IoU 0.773) and other compared methods such as PangNet and DeconvNet. This work
is on a different dataset, so we use it as context rather than a head-to-head
comparison, but it supports the idea that modest changes to the U-Net encoder
can yield real gains. Our choice to swap the vanilla encoder for a ResNet encoder
follows the same spirit, with the added question of whether the gain survives
without pretraining.

Taken together, the three papers point to a clear design: keep the U-Net
decoder and skip connections, strengthen the encoder, handle class imbalance
explicitly, and compare against the FCN numbers from Amgad et al. The
contribution of our report is the controlled separation of encoder depth from
weight initialization, which none of these papers measures directly.

= Dataset

We use the Breast Cancer Semantic Segmentation (BCSS) dataset @amgad2019, which
contains more than twenty thousand region annotations derived from 151 H&E
stained images at 20$times$ magnification. The images are regions of interest
extracted from whole-slide images in the TCGA Breast Invasive Carcinoma
collection, and all cases are triple-negative breast cancer. Annotations were
produced through structured crowdsourcing, where pathologists, residents, and
medical students each labeled a portion of the slides and the harder slides
were assigned to the more senior annotators.

We work with the $512 times 512$ version of the dataset, which provides masks
with 22 fine-grained tissue codes. Following Amgad et al., we group these codes
into five broader classes used for modeling: _tumor_, _stroma_,
_inflammatory_, _necrosis_, and _other_. Codes that do not map to one of the
five classes are sent to an ignore label (value 255) so that they contribute
neither to the loss nor to the metrics.

#figure(
  image("figures/example_input_gt.png", width: 100%),
  caption: [An H&E input patch and its ground-truth mask. Colors denote tumor
  (red), stroma (green), inflammatory (blue), necrosis (yellow), and other
  (gray).],
) <fig:example>

== Descriptive analysis

The dataset is strongly imbalanced. @fig:class_dist shows the pixel
count and the percentage of pixels per class across the three splits. Tumor and
stroma together account for roughly 80% of all labeled pixels, while necrosis
and other occupy only a few percent each. This imbalance is the central
difficulty of the task: a model can reach a high pixel accuracy by predicting
the two dominant classes well and ignoring the rare ones, so per-class metrics
matter more than the overall accuracy.

#figure(
  placement: top,
  image("figures/class_dist.png", width: 88%),
  caption: [Class distribution across the train, validation, and test splits, by
  pixel count (top) and percentage (bottom). Tumor and stroma dominate; necrosis
  and other are rare.],
) <fig:class_dist>

The same imbalance appears at the patch level: necrosis is present in only about
965 of the 6000 original training images, and most patches contain a mix of
classes. The model therefore has to separate neighboring tissue types within an
image rather than classify it as a whole, which is one reason boundary quality
matters for this task.

Because the patches are cropped from large whole-slide images, several patches
can come from the same patient. A random patch-level split would then place
patches from one patient in both training and test, which leaks information and
inflates the reported scores. To avoid this we split by patient: 88 patients
(5111 patches) for training, 18 patients (889 patches) for validation, and the
remaining patients (2768 patches) for a held-out test set. The split uses a
fixed random seed so that it can be reproduced. All numbers reported in this
report are computed on the test patients, which the models never see during
training or model selection.

= Method

== Model variants

All three models follow the same U-Net design: an encoder that reduces spatial
resolution while increasing channel depth, a decoder that upsamples back to full
resolution, and skip connections that join matching encoder and decoder stages.
What changes between the variants is how the encoder is built, and the decoder
widths follow from that choice. @fig:arch shows both: the Vanilla U-Net in panel
(a) and the ResNet-34 variant in panel (b).

#fullwidth[
  #figure(
    {
      scale(40%, reflow: true, draw-network((
        (type: "input", image: image("figures/he_input.png"), channels: ("3", "512"), widths: (0.2,), height: 8, depth: 8, name: "input"),
        (type: "conv", channels: ("64", "512"), widths: (0.4,), height: 8, depth: 8, name: "down1", offset: 1.9),
        (type: "pool", height: 6.5, depth: 6.5, name: "pool1"),
        (type: "conv", channels: ("128", "256"), widths: (0.6,), height: 6.5, depth: 6.5, name: "down2"),
        (type: "pool", height: 5, depth: 5, name: "pool2"),
        (type: "conv", channels: ("256", "128"), widths: (1.0,), height: 5, depth: 5, name: "down3"),
        (type: "pool", height: 3.5, depth: 3.5, name: "pool3"),
        (type: "conv", channels: ("512", "64"), widths: (1.8,), height: 3.5, depth: 3.5, name: "down4"),
        (type: "pool", height: 2.5, depth: 2.5, name: "pool4"),
        (type: "conv", channels: ("1024", "32"), widths: (3.4,), height: 2.5, depth: 2.5, name: "middle"),
        (type: "conv", channels: ("512", "64"), widths: (1.8,), height: 3.5, depth: 3.5, name: "up1", offset: 1.5),
        (type: "conv", channels: ("256", "128"), widths: (1.0,), height: 5, depth: 5, name: "up2", offset: 1.5),
        (type: "conv", channels: ("128", "256"), widths: (0.6,), height: 6.5, depth: 6.5, name: "up3", offset: 1.5),
        (type: "conv", channels: ("64", "512"), widths: (0.4,), height: 8, depth: 8, name: "up4", offset: 1.5),
        (type: "conv", channels: ("5", "512"), widths: (0.2,), height: 8, depth: 8, name: "output"),
      ), connections: (
        (from: "down4", to: "up1", type: "skip", mode: "air", pos: 2.5, touch-layer: true),
        (from: "down3", to: "up2", type: "skip", mode: "air", pos: 3.4, touch-layer: true),
        (from: "down2", to: "up3", type: "skip", mode: "air", pos: 4.1, touch-layer: true),
        (from: "down1", to: "up4", type: "skip", mode: "air", pos: 4.8, touch-layer: true),
      ), show-legend: false))
      text(size: 8pt)[*(a)* Vanilla U-Net]
      v(0.4em)
      scale(40%, reflow: true, draw-network((
        (type: "input", image: image("figures/he_input.png"), channels: ("3", "512"), widths: (0.2,), height: 8, depth: 8, name: "input"),
        (type: "conv", channels: ("64", "256"), widths: (0.4,), height: 6.5, depth: 6.5, name: "e1", offset: 1.9),
        (type: "pool", height: 5, depth: 5, name: "pool"),
        (type: "conv", channels: ("64", "128"), widths: (0.4,), height: 5, depth: 5, name: "e2"),
        (type: "conv", channels: ("128", "64"), widths: (0.6,), height: 3.5, depth: 3.5, name: "e3"),
        (type: "conv", channels: ("256", "32"), widths: (1.0,), height: 2.5, depth: 2.5, name: "e4"),
        (type: "conv", channels: ("512", "16"), widths: (1.8,), height: 1.8, depth: 1.8, name: "e5"),
        (type: "conv", channels: ("256", "32"), widths: (1.0,), height: 2.5, depth: 2.5, name: "d5", offset: 1.5),
        (type: "conv", channels: ("128", "64"), widths: (0.6,), height: 3.5, depth: 3.5, name: "d4", offset: 1.5),
        (type: "conv", channels: ("64", "128"), widths: (0.4,), height: 5, depth: 5, name: "d3", offset: 1.5),
        (type: "conv", channels: ("64", "256"), widths: (0.4,), height: 6.5, depth: 6.5, name: "d2", offset: 1.5),
        (type: "conv", channels: ("32", "512"), widths: (0.3,), height: 8, depth: 8, name: "d1", offset: 1.5),
        (type: "conv", channels: ("5", "512"), widths: (0.2,), height: 8, depth: 8, name: "output"),
      ), connections: (
        (from: "e4", to: "d5", type: "skip", mode: "air", pos: 2.5, touch-layer: true),
        (from: "e3", to: "d4", type: "skip", mode: "air", pos: 3.4, touch-layer: true),
        (from: "e2", to: "d3", type: "skip", mode: "air", pos: 4.1, touch-layer: true),
        (from: "e1", to: "d2", type: "skip", mode: "air", pos: 4.8, touch-layer: true),
      ), show-legend: false))
      text(size: 8pt)[*(b)* ResNet-34 U-Net]
    },
    caption: [The two U-Net architectures compared in this study; channels are
    annotated below each block and spatial size is shown by block height. *(a)*
    Vanilla U-Net: four double-convolution blocks contract the H&E input to a
    1024-channel bottleneck and a symmetric transposed-convolution decoder
    restores full resolution. *(b)* ResNet-34 U-Net: a ResNet-34 encoder (a
    $7 times 7$ stride-2 stem and max-pool followed by four residual stages)
    gives five downsampling levels and a 512-channel bottleneck, and the decoder
    reuses the same transposed-convolution blocks at channel widths matched to
    the ResNet features. Both variants share the decoder design and $1 times 1$
    output head and differ in the encoder; skip connections join matching encoder
    and decoder stages.],
  ) <fig:arch>
]

The *Vanilla U-Net* uses the classic encoder of four double-convolution blocks
with max-pooling, a bottleneck, and a symmetric decoder built from transposed
convolutions. Each convolution block applies two $3 times 3$ convolutions with
batch normalization and ReLU, and dropout is added in the deeper blocks to
reduce overfitting.

The *ResNet-34 U-Net* (@fig:arch (b)) replaces the encoder with the four
residual stages of a ResNet-34, preceded by its $7 times 7$ stride-2 stem and a
max-pool, so the encoder has five downsampling levels and a 512-channel
bottleneck rather than the Vanilla U-Net's four levels and 1024 channels. The
decoder keeps the same building block as the Vanilla U-Net, a transposed
convolution followed by a double convolution at each stage, but its channel
widths are matched to the narrower ResNet feature maps (256, 128, 64, and 64
from the bottleneck outward), and a final upsampling block returns to full
resolution before the shared $1 times 1$ output head. We build two versions of
this network: one initialized with random weights, and one initialized with
ImageNet-pretrained weights from torchvision. The ResNet-34 variant has fewer
parameters than the Vanilla U-Net, 24.45 million against 31.04 million, so any
improvement is not simply the result of a larger model: the encoder is better
designed, not bigger.

== Loss and class imbalance

We train with a combined loss that adds class-weighted cross-entropy to a soft
Dice loss. Cross-entropy provides a stable pixel-wise gradient, while the Dice
term directly optimizes region overlap and is less sensitive to imbalance. Class
weights computed from the training distribution further raise the importance of
the rare classes, and a small amount of label smoothing (0.05) discourages
over-confident predictions. The ignore label is excluded from both terms so that
unmapped tissue codes do not affect the optimization.

== Data augmentation

Histopathology slides vary in color because of differences in staining and
scanning, which shows up as a wide per-channel intensity spread across the
training set. Rather than apply explicit color normalization, we handle this
variability with augmentation using the Albumentations library @buslaev2020. The
pipeline combines geometric transforms (random flips and rotations) with
pixel-level transforms (brightness and contrast changes, blur, and
hue-saturation shifts) so that the model sees a wide range of appearances during
training. @fig:aug shows several augmented versions of the same patch.

#figure(
  image("figures/aug_montage.png", width: 78%),
  caption: [Augmented views of training patches produced by the Albumentations
  pipeline. Geometric and color transforms expand the effective range of
  appearances the model sees.],
) <fig:aug>

== Hyperparameter optimization

We selected the main training hyperparameters with Optuna @akiba2019 using a
tree-structured Parzen estimator sampler and a median pruner that stops weak
trials early. The search ran for 25 trials on the ResNet-34 encoder and tuned
the learning rate (log-uniform in $[10^(-5), 10^(-2)]$), the weight decay
(log-uniform in $[10^(-6), 10^(-2)]$), the batch size (8 or 16), the optimizer
(Adam or AdamW), and the loss (cross-entropy or the cross-entropy plus Dice
combination). Each trial trained for a reduced number of epochs and was scored
by mean validation Dice. The best configuration used Adam with a learning rate
near $9.2 times 10^(-5)$, a batch size of 16, and the combined cross-entropy
plus Dice loss. We reused this configuration for all three variants so that the
comparison stays controlled.

= Experimental Setup

Each model was trained for 50 epochs with the Adam optimizer and the tuned
learning rate. The schedule applies a short linear warmup over the first five
epochs followed by cosine annealing for the rest of training. We use dropout in
the decoder, weight decay, and label smoothing as regularizers, and mixed
precision to fit the $512 times 512$ inputs into GPU memory. Training was run on
a single Kaggle GPU. We track the training and validation loss together with
mean Dice, mean IoU, and pixel accuracy after every epoch, and we keep the
checkpoint with the best validation Dice for the final test evaluation.

Metric aggregation matters for this task. We accumulate Dice and IoU as raw
per-class pixel counts (intersections and unions) over the whole loader and
convert them to a ratio only at the end of each epoch. This global, count-based
aggregation is independent of batch size, whereas averaging per-batch ratios
would bias the score on the rare classes that are absent from many batches. All
numbers in this report use the global metric.

We report three standard metrics. The Dice coefficient measures the overlap
between prediction and ground truth for each class and is our main metric, IoU
is a stricter overlap measure, and pixel accuracy is the fraction of correctly
labeled pixels. We average Dice and IoU over the five classes with macro
averaging so that the rare classes count as much as the dominant ones.

= Results and Discussion

== Training behavior

@fig:train shows the training and validation loss (a) and the validation Dice
and IoU (b) for all three models over the 50 epochs. All
three models converge, but the two ResNet variants stay above the Vanilla U-Net
for the whole training run, and the pretrained model reaches the lowest loss.
The pretrained encoder also converges faster in the early epochs, which is the
expected effect of starting from useful ImageNet features rather than random
weights. The gap between training and validation loss stays moderate, which
suggests the regularization kept overfitting under control.

#fullwidth[
  #figure(
    {
      image("figures/loss_curves.png", width: 60%)
      text(size: 8pt)[*(a)* Training and validation loss]
      v(0.5em)
      image("figures/valid_metric_curves.png", width: 60%)
      text(size: 8pt)[*(b)* Validation Dice and IoU]
    },
    caption: [Training behavior over 50 epochs. *(a)* Training and validation
    loss (cross-entropy plus Dice); the pretrained ResNet-34 model reaches the
    lowest loss. *(b)* Validation Dice and IoU; both ResNet variants remain above
    the Vanilla U-Net throughout.],
  ) <fig:train>
]

== Quantitative comparison

@tab:overall reports the overall metrics on the held-out test set
alongside the FCN baseline from Amgad et al. @amgad2019. The Vanilla U-Net reaches
a mean Dice of 0.760, which essentially matches the baseline. Swapping in the
ResNet-34 encoder with random weights lifts the mean Dice to 0.809, and adding
ImageNet pretraining lifts it further to 0.824. The same ordering holds for mean
IoU and pixel accuracy: the pretrained model is best on every overall metric.

#figure(
  placement: top,
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    inset: (x: 5pt, y: 3pt),
    table.hline(),
    table.header(
      [*Model*], [*mDice*], [*mIoU*], [*Pixel acc.*],
    ),
    table.hline(stroke: 0.5pt),
    [FCN baseline @amgad2019], [$tilde.op$0.75], [$tilde.op$0.61], [---],
    [Vanilla U-Net], [0.760], [0.622], [0.806],
    [ResNet-34 (random)], [0.809], [0.683], [0.849],
    [ResNet-34 (pretrained)], [*0.824*], [*0.704*], [*0.857*],
    table.hline(),
  ),
  caption: [Overall test-set metrics (macro-averaged Dice and IoU, pixel
  accuracy). The FCN row is the literature baseline reported on the same dataset
  and is not an exact head-to-head split.],
) <tab:overall>

This two-step comparison is effectively an ablation. With the decoder, loss,
augmentation, data split, and hyperparameters all held fixed, we change one
factor at a time, first the encoder and then its initialization, so each Dice
gain can be attributed to a single design choice. The encoder change alone (vanilla
to random ResNet) adds about 0.05 mean Dice, while pretraining adds a further
0.015. Most of the improvement therefore comes from the stronger encoder
architecture, and ImageNet initialization contributes a smaller but consistent
extra gain. That ordering is reassuring when natural images and histopathology
look so different, because it suggests the encoder design transfers more reliably
than the exact pretrained features.

@tab:perclass breaks the test Dice down by class. The dominant classes
(tumor and stroma) are already handled well by every model, so the headline
gains come from the rarer classes. Necrosis is the clearest case: its Dice rises
from 0.588 with the Vanilla U-Net to 0.832 with the pretrained model, a gain of
about 0.24. Inflammatory tissue improves in a similar way, so the gap between the
easy and hard classes, while still present for the best model, is much smaller
than for the Vanilla U-Net.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    inset: (x: 4pt, y: 3pt),
    table.hline(),
    table.header(
      [*Class*], [*Vanilla*], [*RN34 rand.*], [*RN34 pre.*],
    ),
    table.hline(stroke: 0.5pt),
    [tumor],        [0.878], [0.896], [*0.901*],
    [stroma],       [0.791], [0.839], [*0.847*],
    [inflammatory], [0.744], [0.778], [*0.778*],
    [necrosis],     [0.588], [0.800], [*0.832*],
    [other],        [0.797], [0.733], [0.763],
    table.hline(stroke: 0.5pt),
    [*mean*],       [0.760], [0.809], [*0.824*],
    table.hline(),
  ),
  caption: [Per-class test Dice for the three variants. The largest improvement
  is on necrosis, the rarest class.],
) <tab:perclass>

It is worth separating ranking quality from the operating point. The
macro-averaged area under the ROC curve, computed from the predicted
probabilities on a large sample of test pixels, lands near 0.96 for all three
models (0.965, 0.965, and 0.963 for vanilla, random, and pretrained), so the
probabilities already rank pixels well even for the Vanilla U-Net. The Dice gap
therefore comes mostly from how cleanly the argmax turns those probabilities into
a mask, especially on the rare classes and near boundaries, rather than from any
failure to discriminate the tissue types.

== Qualitative results

@fig:qual shows predictions for two test patches together with the input
and the ground truth. The Vanilla U-Net produces noisier masks with scattered
misclassified pixels, especially around region boundaries. The two ResNet models
produce cleaner masks that follow the tissue borders more closely, and the
pretrained model is the most consistent of the three. This matches the
quantitative ordering and confirms that the numerical gain corresponds to
visibly better segmentation rather than a metric artefact.

#fullwidth[
  #figure(
    image("figures/qualitative.png", width: 76%),
    caption: [Qualitative predictions on two test patches. From left to right:
    H&E input, ground truth, Vanilla U-Net, ResNet-34 (random), and ResNet-34
    (pretrained). The per-patch Dice is printed on each prediction. The ResNet
    models give cleaner masks that follow the tissue boundaries.],
  ) <fig:qual>
]

== Error analysis

@fig:cm is the row-normalized confusion matrix of the best model. The
diagonal is strong, with most classes correctly labeled at least 80% of the
time. The main off-diagonal mass is the confusion between stroma and
inflammatory tissue: about 17% of true inflammatory pixels are predicted as
stroma. This is an understandable error, because inflammatory infiltrate and
stroma can look similar in H&E and often appear next to each other. The
remaining errors are small and mostly involve the boundary between tumor and
stroma. The model rarely confuses necrosis or the other class once trained,
which is consistent with the recovered per-class Dice for necrosis.

#fullwidth[
  #figure(
    image("figures/confusion_matrix.png", width: 44%),
    caption: [Row-normalized confusion matrix for the pretrained ResNet-34
    U-Net. The main confusion is between stroma and inflammatory tissue.],
  ) <fig:cm>
]

Across individual test patches rather than the pooled score, the best model's
per-image mean Dice has a median of 0.72 and most patches score above 0.7, with a
clear left tail of difficult patches below 0.4. This per-image average is lower
than the pixel-pooled value in @tab:overall because every patch counts equally,
so a few small or ambiguous patches pull the mean down even when most pixels are
segmented well. The tail is where the remaining errors live.

== External generalisation

The held-out test set still comes from the same archive (TCGA) as the training
data, so we also checked how the models behave on slides from other sources. We
use two external datasets: the TIGER challenge tissue regions @tiger2022 for a
quantitative test, and the TNBC dataset @naylor2019 (the data used by Drioua et
al. @drioua2023) for a qualitative one.

TIGER provides region-of-interest masks at 20$times$ with a seven-class tissue
taxonomy that we map onto our five classes (invasive and in-situ tumor to tumor,
tumor-associated stroma to stroma, inflamed stroma to inflammatory, necrosis to
necrosis, and healthy glands and the rest class to other). Its ROIs come from two
kinds of slides: some from TCGA, the same archive BCSS was built from, and some
from two other hospitals (Radboud and Jeroen Bosch). We score these groups
separately, because only the non-TCGA slides are a genuinely external test with
no overlap with our training patients. Each ROI is tiled into 512$times$512
windows and scored with the same global count-based metrics, ignoring the exclude
label. @tab:external reports the result next to the in-domain BCSS score.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    inset: (x: 3.5pt, y: 3pt),
    table.hline(),
    table.header([*Test set (mDice)*], [*Vanilla*], [*RN34 r.*], [*RN34 pre.*]),
    table.hline(stroke: 0.5pt),
    [BCSS (in-domain)],        [0.760], [0.809], [*0.824*],
    [TIGER (TCGA)],            [0.357], [0.334], [*0.490*],
    [TIGER (external)],        [0.333], [0.322], [*0.371*],
    table.hline(),
  ),
  caption: [Macro Dice on the external TIGER ROIs versus in-domain BCSS. The
  pretrained model is best everywhere; scores drop sharply off-domain, and more
  so on the truly external (non-TCGA) slides.],
) <tab:external>

These numbers are worth reading carefully. The in-domain ranking is preserved on
every split, and the pretrained ResNet-34 still generalises best, so the main
finding of this report holds out of distribution. At the same time there is a
large domain gap: macro Dice falls from about 0.82 on BCSS to about 0.37 on the
truly external slides, and the same-source TCGA slides sit in between. Part of this drop
is a real shift in scanner and preparation, and part is an imperfect label
correspondence (TIGER's healthy and rest tissue does not match our other class
exactly), so the ranking and the size of the gap are the takeaway rather than the
absolute values.

@fig:external shows predictions on five genuinely external (non-TCGA) TIGER ROIs
next to their ground truth. The visual story matches the table. All three models
drift on the unfamiliar slides, and in particular over-predict necrosis (yellow)
where the reference is tumor or stroma, but the pretrained ResNet-34 stays closest
to the ground truth and keeps the large tumor and stroma regions intact. We
observed the same qualitative behavior on the TNBC dataset of Drioua et al.
@naylor2019 @drioua2023, which is labeled only at the nucleus level and so
cannot be scored against our tissue classes.

#fullwidth[
  #figure(
    image("figures/external_tiger.png", width: 64%),
    caption: [Qualitative predictions on five external TIGER ROIs (Radboud and
    Jeroen Bosch slides, no overlap with training). Left to right: H&E input,
    ground truth, and the three models. The pretrained ResNet-34 tracks the
    ground truth most closely; all models over-predict necrosis under the domain
    shift.],
  ) <fig:external>
]

== Discussion

The experiment supports a simple takeaway. A deeper, well-designed encoder is
the larger lever for this task, and transfer learning adds a smaller, dependable
improvement on top, even with fewer parameters than the Vanilla U-Net. The
biggest practical benefit shows up exactly where it is most needed, on the rare
classes that the Vanilla U-Net struggles with. The remaining weakness, the stroma
versus inflammatory confusion, is a genuine visual ambiguity rather than an
obvious modeling flaw, and it is the most natural target for future work, for
example through stain-aware features or a loss that penalizes this specific
confusion more heavily.

Two limitations should be stated plainly. First, the FCN baseline is taken from
the Amgad et al. paper and is measured on that paper's own split, so the
comparison is a literature reference and not an exact head-to-head. Second, all
results are on the BCSS triple-negative cohort, so the conclusions may not carry
over unchanged to other breast cancer subtypes or other staining protocols.

== Implementation and reproducibility

The project is implemented in PyTorch. The model definitions, dataset loader,
training loop, loss functions, and evaluation code live in small reusable modules,
and the experiments run from Jupyter notebooks that keep their cell outputs, so the
processed data, the network definition, the loss curves, and the final metrics can
be inspected without rerunning anything. One notebook drives the Optuna search and
one notebook per variant trains and evaluates the model, all under a single fixed
seed so the patient-level split is identical across runs. We do not ship the
trained weights or the dataset, since both are large; the dataset is the public
BCSS release and can be downloaded from its original location. These materials
accompany this report as the code deliverable.

= Conclusion

We trained and compared three U-Net variants for five-class breast cancer tissue
segmentation on the BCSS dataset under a controlled, leakage-free setup. Moving
from a Vanilla U-Net to a ResNet-34 encoder raised the mean test Dice from 0.760
to 0.809, and adding ImageNet pretraining raised it further to 0.824, with all
three models at or above the FCN baseline of roughly 0.75. The two-step design
let us attribute most of the gain to the encoder architecture and a smaller,
consistent part to transfer learning. The clearest improvement was on the rarest
class, necrosis, whose Dice climbed from 0.59 to 0.83. The main remaining error
is the visually justified confusion between stroma and inflammatory tissue,
which we leave as the most promising direction for future work. All code,
notebooks with preserved outputs, and the figures used here are provided with
this report.

// start the reference list at the top of the (right) column as one block
#colbreak()
#show bibliography: set text(size: 8pt)
#bibliography("references.yml", title: "References", style: "ieee")
