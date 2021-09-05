# DSP HW4 
<center> <font size=2> Jason [2020/12] </font> </center>

original hackmd : https://hackmd.io/@jasonho610/HyI4Wcksw

## What is Objective Multichannel Loudness Measurement Algorithm?
“Loudness” is a subjective perception of how loud a sound sounds to the human ear. Loudness meters calculate the loudness value of a program sound by filtering it on the basis of the properties of the human ear and weighting it according to the direction of the audio channels. 
Broadcasters can now use loudness meters to normalize loudness values across all programs.

The algorithm consists of four stages:

![](https://i.imgur.com/8snGpxN.png)

- “K” frequency weighting;
    - An electrical filter which is designed to mimic the relative sensitivity of the human ear to different frequencies in terms of perceived loudness.
- mean square calculation for each channel;
- channel-weighted summation (surround channels have larger weights, and the LFE channel is excluded);
- gating of 400 ms blocks.

In HW4, we foucus on the rebuilding of first stage, K-weighted frequency filter in sample frequency $f_s=48$ kHz (the coefficients of $H(z)$ are given in spec), and also find out what the filter coefficients are, when $f_s = 20$ kHz.

## K-weighted Frequency Filter
The K-weighting filter is composed of two stages of filtering; a first stage shelving filter and a second stage high-pass filter. The first stage of the pre-filtering accounts for the acoustic effects of the head, where the head is modelled as a rigid sphere.

**1st Stage Filter properties:**
| <font size=2>**Freqency response**</font> | <font size=2>**Signal flow & Filter coefficients**</font> |
| :--------: | :--------: |
| ![mag of stage1](https://i.imgur.com/U16zC6y.png =400x) | ![sigflow of stage1](https://i.imgur.com/5u2XN6v.png =400x)<br> ![coeff of stage1](https://i.imgur.com/ZPWkXAk.png =400x)|

The second stage of the pre-filter applies a simple high-pass filter.

**2nd Stage Filter properties:**
| <font size=2>**Freqency response**</font> | <font size=2>**Signal flow & Filter coefficients**</font> |
| :--------: | :--------: |
| ![mag of stage2](https://i.imgur.com/EXQyUUh.png =400x) | ![sigflow of stage2](https://i.imgur.com/lTeLHa1.png =400x)<br> ![coeff of stage2](https://i.imgur.com/ATIJVyy.png =400x) |

Together, the signal flow of K-weighted filter is shown as follows:

![sigflow of Kw](https://i.imgur.com/7s6DYj0.png =550x)

:::warning
<font size = 2>These filter coefficients are for a sampling rate of 48 kHz. Implementations at other sampling rates will require different coefficient values, which should be chosen to provide the same frequency response that the specified filter provides at 48 kHz.</font>
:::

## Code Demonstration
This python program has several objectives:
- Representing $\lvert H(e^{j\omega})\rvert$, $\measuredangle H(e^{j\omega})$, $grd[H(e^{j\omega})]$ of 1st stage filter, 2nd stage filter and K-weighted filter respectively
- Analysing the filters by plotting $H(z)$ on z-plane
- Looking for coefficients specified for 20kHz

### Libraries
```python
import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import warnings
warnings.filterwarnings("ignore")
```

### Plotting Functions
```python
def plotMag(omega, H, title, xlim, ylim):
    plt.figure(figsize = (11,9))
    H_db = H_db = 20*np.log10(np.abs(H))
    plt.semilogx(omega, H_db, color = "black", linewidth = 2)
    plt.xlim(xlim)
    plt.xlabel("Frequency (Hz)", fontsize = 12)
    plt.ylim(ylim)
    plt.ylabel("Relative level (dB)", fontsize = 12)
    plt.grid(True, which = "both", color = "black", linestyle = (0, (5, 10)))
    plt.title(title + '\n', fontsize = 16)
    plt.savefig(title + ".pdf")
    plt.show()
```
```python
def plotPhs(omega, H, title, xlim):
    plt.figure(figsize = (11,9))
    H_ph = np.rad2deg(np.unwrap(np.angle(H)))
    plt.semilogx(omega, H_ph, color = "black", linewidth = 2)
    plt.xlim(xlim)
    plt.xlabel("Frequency (Hz)", fontsize = 12)
    plt.ylabel("Phase (degree)", fontsize = 12)
    plt.grid(True, which = "both", color = "black", linestyle = (0, (5, 10)))
    plt.title(title + '\n', fontsize = 16)
    plt.savefig(title + ".pdf")
    plt.show()
```
```python
def plotGD(omega, gd, title, xlim):
    plt.figure(figsize = (11,9))
    plt.semilogx(omega, gd, color = "black", linewidth = 2)
    plt.xlim(xlim)
    plt.xlabel("Frequency (Hz)", fontsize = 12)
    plt.ylabel("Group Delay", fontsize = 12)
    plt.grid(True, which = "both", color = "black", linestyle = (0, (5, 10)))
    plt.title(title + '\n', fontsize = 16)
    plt.savefig(title + ".pdf")
    plt.show()
```
```python
def ZPlane(z, p, title):
    figure, ax = plt.subplots(figsize = (5, 5))
    uc = plt.Circle((0, 0), 1, color = "black", fill = False)
    ax.add_patch(uc)
    
    plt.scatter(z.real, z.imag, label = "zero", marker = 'o', facecolors = "none", edgecolors='royalblue')
    plt.scatter(p.real, p.imag, label = "pole", marker = 'x', facecolors = "darkorange")
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.xlabel("$\mathfrak{Re}$")
    plt.ylabel("$\mathfrak{Im}$")
    plt.legend()
    
    plt.axhline(0, color = "black", linewidth = 1)
    plt.axvline(0, color = "black", linewidth = 1)
    plt.title(title)
    plt.savefig(title + ".pdf")
    plt.show()
```

### 1st Stage Filter
```python
f_s = 48000
a1 = [1.0, -1.69065929318241, 0.73248077421585]
b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
omega, H1 = ss.freqz(b1, a1, worN = 10000, fs = f_s)
plotMag(omega, H1, "Response of stage 1", xlim = [10, 20000], ylim = [-10, 10])
plotPhs(omega, H1, "Phase of stage 1", xlim = [10, 20000])
omega, gd1 = ss.group_delay((b1, a1), w = 10000, fs = f_s)
plotGD(omega, gd1, "Group Delay of stage 1", xlim = [10, 20000])
```

![](https://i.imgur.com/nLXzEU1.png =350x) ![](https://i.imgur.com/5zV0kHe.png =350x)![](https://i.imgur.com/xelfjxy.png =350x)


### 2nd Stage Filter
```python
a2 = [1.0, -1.99004745483398, 0.99007225036621]
b2 = [1.0, -2.0, 1.0]
omega, H2 = ss.freqz(b2, a2, worN = 10000, fs = f_s)
plotMag(omega, H2, "Response of stage 2", xlim = [10, 20000], ylim = [-30, 5])
plotPhs(omega, H2, "Phase of stage 2", xlim = [10, 20000])
omega, gd2 = ss.group_delay((b2, a2), w = 10000, fs = f_s)
plotGD(omega, gd2, "Group Delay of stage 2", xlim = [10, 20000])
```
![](https://i.imgur.com/s990xYv.png =350x)![](https://i.imgur.com/vQpM84R.png =350x)![](https://i.imgur.com/euiTdss.png =350x)


### K-weighted Filter

```python
a_kfilter = np.convolve(a1, a2)
b_kfilter = np.convolve(b1, b2)
omega, H_kfilter = ss.freqz(b_kfilter, a_kfilter, worN = 10000, fs = f_s)
plotMag(omega, H_kfilter, "Response of Kw_filter", xlim = [10, 20000], ylim = [-30, 10])
plotPhs(omega, H_kfilter, "Phase of Kw_filter", xlim = [10, 20000])
omega, gd_kfilter = ss.group_delay((b_kfilter, a_kfilter), w = 10000, fs = f_s)
plotGD(omega, gd_kfilter, "Group Delay of Kw_filter", xlim = [10, 20000])
```

|  | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| **b_kfilter** |1.53512486|-5.76194591|8.11691005|-5.08848181|1.19839281
| **a_kfilter** |1.0|-3.68070675|5.08704525|-3.13154635|0.72520889|

![](https://i.imgur.com/82PKmbH.png =350x)![](https://i.imgur.com/XUt0DJH.png =350x)![](https://i.imgur.com/ZZYOXcr.png =350x)



### Poles and Zeros
```python
z1 = np.roots(b1)
p1 = np.roots(a1)
z2 = np.roots(b2)
p2 = np.roots(a2)
z_kfilter = np.roots(b_kfilter)
p_kfilter = np.roots(a_kfilter)

ZPlane(z1, p1, "ZPlane of stage 1")
ZPlane(z2, p2, "ZPlane of stage 2")
ZPlane(z_kfilter, p_kfilter, "ZPlane of Kw_filter")
```

|  | 1 | 2 |
| --- | --- | --- |
| **z_1** |0.87670269+0.10973068j|0.87670269-0.10973068j|
| **p_1** |0.84532965+0.13378551j|0.84532965-0.13378551j|
| **z_2** |1+0j|1+0j|
| **p_2** |0.99502373+0.00017956j|0.99502373-0.00017956j|



|  | 1 | 2 | 3 | 4 |
| ---  | --- | --- | --- | --- |
| **z_kfilter** |1.00000023+0j|0.99999977+0j|0.87670269+0.10973068j|0.87670269-0.10973068j|
| **p_kfilter** |0.99502373+0.00017956j|0.99502373-0.00017956j|0.84532965+0.13378551j|0.84532965-0.13378551j|

![](https://i.imgur.com/2rUBhwt.png =240x)![](https://i.imgur.com/Q5qXp5v.png =240x)![](https://i.imgur.com/mlUlAwG.png =240x)


### Finding Coeffs for 20kHz

Each zero and pole has their corresponding frequency to act on; our goal is to find them out and re-plot it from freqency scale $[0\to 48k]$ to $[0\to 20k]$.


```python
z_r = np.absolute(z_kfilter)
z_theta = np.angle(z_kfilter)
p_r = np.absolute(p_kfilter)
p_theta = np.angle(p_kfilter)
# coordinate to polar

z_freq = 48000.0 * z_theta / 20000.0
p_freq = 48000.0 * p_theta / 20000.0

z_new = []; p_new = []
for i in range(4):
    z_new.append(z_r[i] * complex(np.cos(z_freq[i]), np.sin(z_freq[i])))
    p_new.append(p_r[i] * complex(np.cos(p_freq[i]), np.sin(p_freq[i])))
z_new = np.array(z_new); p_new = np.array(p_new)
```

|  | 1 | 2 | 3 | 4 |
| ---  | --- | --- | --- | --- |
| **z_new** |1.00000023+0j|0.99999977+0j|0.84438407+0.260123j|0.84438407-0.260123j|
| **p_new** |0.99502365+0.00043095j|0.99502365-0.00043095j|0.79583863+0.3148359j|0.79583863-0.3148359j|



```python
ZPlane(z_new, p_new, "ZPlane of Kw_filter (in 20Hz)")
```
![](https://i.imgur.com/JPhuIP1.png =300x)

```python
from sympy import *
from IPython.display import Math

z = Symbol('z')
Y_new = S(1); X_new = S(1)
for i in range(4):
    Y_new = Y_new * (z-z_new[i])
    X_new = X_new * (z-p_new[i])

Y_new = Y_new.expand()
X_new = X_new.expand()
H_new = (Y_new / X_new).simplify()

display(Math("Y(z) = "+latex(Y_new.expand())))
display(Math("X(z) = "+latex(X_new.expand())))
display(Math("H(z) = "+latex(H_new)))
```
$\displaystyle Y(z) = z^{4} - 3.68876813577454 z^{3} + 5.1581847011337 z^{2} - 3.25006499494377 z + 0.7806484295846$

$\displaystyle X(z) = z^{4} - 3.58172456808702 z^{3} + 4.89006607417582 z^{2} - 3.03354688155039 z + 0.725208888477869$

$\displaystyle H(z) = \frac{z^{4} - 3.68876813577454 z^{3} + 5.1581847011337 z^{2} - 3.25006499494377 z + 0.7806484295846}{z^{4} - 3.58172456808702 z^{3} + 4.89006607417582 z^{2} - 3.03354688155039 z + 0.725208888477869}$

### K-weighted filter in 20 kHz
```python
b_new = []; a_new = []
for i, j in zip(Poly(Y_new).coeffs(), Poly(X_new).coeffs()):
    b_new.append(float(i)); a_new.append(float(j))
```
|  | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| **b_new** |1.0|-3.688768135774538|5.158184701133702|-3.250064994943768|0.7806484295845998
| **a_new** |1.0|-3.5817245680870187|4.890066074175821|-3.033546881550394|0.725208888477869|

```python
f_s_new = 20000
omega_new, H_new = ss.freqz(b_new, a_new, worN = 10000, fs = f_s_new)
plotMag(omega_new, H_new, "Response of Kw_filter (in 20Hz)", xlim = [10, 20000], ylim = [-30, 10])
plotPhs(omega_new, H_new, "Phase of Kw_filter (in 20Hz)", xlim = [10, 20000])
omega_new, gd__new = ss.group_delay((b_new, a_new), w = 10000, fs = f_s_new)
plotGD(omega_new, gd__new, "Group Delay of Kw_filter (in 20Hz)", xlim = [10, 20000])
```
![](https://i.imgur.com/jgneuZW.png =350x)![](https://i.imgur.com/NXT3UxQ.png =350x)![](https://i.imgur.com/hVkrMvL.png =350x)

Unfortunately, this method seems to be naive; I've tried to adjust the position of poles & zeros in term of two parameters:
- norm = $\sqrt{x^2+y^2}$, i.e, magnitude = $r$
- argument = $tan^{-1}(x/y)$, i.e, phase = $\theta$

to get any closer result like the frequency response of k-weighted filter.
Nevertheless, "A small leak will sink the great ship," any slight change can ruin the whole scheme. It is nearly possible to get desired consequence by try and error.


<font size = 2>(I've tried my best... ><) </font>
