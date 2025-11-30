Equations (7) and (8) of Buffet et al. 2016 (Geophys. J Int.), followed by the definitions of the symbols.

### Equations (7) and (8)

$\partial_t \mathbf{v} + 2\hat{\mathbf{z}} \times \mathbf{v} = -\nabla P + E_\eta B_r \partial_r \mathbf{b} - \tilde{N}^2 (\mathbf{u} \cdot \hat{\mathbf{r}})\hat{\mathbf{r}} + E\nabla^2 \mathbf{v} $

$\partial_t \mathbf{b} = B_r \partial_r \mathbf{v} + E_\eta \nabla^2 \mathbf{b} $

### Symbol Definitions

The equations are presented in non-dimensional form. The definitions and scalings provided in the text are:

*   **$\mathbf{v}$**: The velocity perturbation vector.
*   **$\mathbf{b}$**: The magnetic field perturbation vector.
*   **$\mathbf{u}$**: The displacement vector associated with the wave (where $\partial_t \mathbf{u} \approx \mathbf{v}$).
*   **$t$**: Time (scaled by $\Omega^{-1}$).
*   $r$: radial coordinate
*   **$\hat{\mathbf{z}}$**: The unit vector along the rotation axis (associated with the mean rotation rate $\mathbf{\Omega} = \Omega \hat{\mathbf{z}}$).
*   **$P$**: The pressure field perturbation (specifically related to the horizontal pressure gradient).
*   **$B_r$**: The radial component of the background steady magnetic field.
*   **$\hat{\mathbf{r}}$**: The unit vector in the radial direction.
*   **$\tilde{N}^2$**: The squared dimensionless buoyancy frequency, defined as $\tilde{N}^2 = N^2 / \Omega^2$ (where $N$ is the buoyancy frequency).
*   **$E$**: The Ekman number, defined as $E = \nu / \Omega R^2$ (representing viscous effects).
*   **$E_\eta$**: The magnetic Ekman number, defined as $E_\eta = \eta / \Omega R^2$ (related to magnetic diffusivity).

**Scaling Factors:**
*   **Lengthscale**: Radius of the core, $R$.
*   **Timescale**: Inverse of the rotation rate, $\Omega^{-1}$.
*   **Magnetic Field Scale**: $\sqrt{\Omega \rho \mu \eta}$.

**Physical Constants:**
*   **$\Omega$**: Mean rotation rate of the Earth.
*   **$\rho$**: Fluid density.
*   **$\nu$**: Fluid viscosity.
*   **$\eta$**: Magnetic diffusivity (where $\eta = (\mu \sigma)^{-1}$).
*   **$\mu$**: Magnetic permeability.
*   **$\sigma$**: Electrical conductivity.



### IMPLEMENTATION PLAN:

We'll use a spectral representation in terms of fully normalized surface spherical harmonics, $Y_\ell^m(\theta,\varphi)$, where $\theta$ is colatitude, and $\varphi$ is longitude:

$$\langle Y_{\ell'}^{m'},Y_\ell^m\rangle = \int_0^{2\pi} \int_0^\pi Y_\ell^m {Y_{\ell'}^{m'}}^\ast \sin\theta d\theta d\varphi = \delta_{\ell',\ell}\delta_{m',m}$$

These are eigenfunctions of the Laplace operator on a sphere surface:

$$\Delta_H Y_{\ell}^m = \ell(\ell+1) Y_{\ell}^m$$

where:

$$\Delta_H \tilde u_r = -\frac{1}{\sin\theta} \frac{\partial}{\partial \theta} \left(\sin\theta \frac{\partial \tilde u_r}{\partial \theta}\right) - \frac{1}{\sin^2\theta}\frac{\partial^2 \tilde u_r}{\partial \varphi^2}$$



#### Part I Neutral buoyancy ($N\equiv 0$) and no magnetic field: 

1) $i \omega \tilde{\mathbf{v}} + 2\hat{\mathbf{z}} \times \tilde{\mathbf{v}} = -\nabla \tilde{P} + E\nabla^2 \tilde{\mathbf{v}}$

2) $\nabla \cdot \tilde{\mathbf{v}} = 0$

(Greenspan's classic intertial modes)

Poloidal-toroidal representation (ensures 2. is satisfied):

$$\tilde{\mathbf{v}} = \sum_{{\ell'},{m'}}\left(\nabla \times( \nabla \times (W_{\ell'}^{m'}(r) Y_{\ell'}^{m'}(\theta,\varphi)\hat{\mathbf{r}})) + \nabla \times (Z_{\ell'}^{m'}(r) Y_{\ell'}^{m'}(\theta,\varphi)\hat{\mathbf{r}})\right)$$

1) $\left\langle\hat{r}\cdot\nabla\times 1.,Y_\ell^m\right\rangle$:

$$A_\ell^m Z_\ell^m = C_\ell^m W_{\ell-1}^m + D_\ell^m W_{\ell+1}^m$$

2) $\left\langle \hat{r} \cdot \nabla \times \nabla \times 1.,Y_\ell^m\right\rangle$:
$$A_\ell^m B_\ell^m W_\ell^m = C_\ell^m Z_{\ell-1}^m +D_\ell^m Z_{\ell+1}^m$$

Where:

$A_\ell^m = i(\ell(\ell+1)\omega -2m) + \ell(\ell+1)EB_\ell^m$
$B_\ell^m = \frac{\ell(\ell+1)}{r^2} - \dfrac{d^2}{dr^2}$
$C_\ell^m = 2(\ell-1)(\ell+1) \sqrt{\frac{(\ell-m)(\ell+m)}{(2\ell-1)(2\ell+1)}} \left(\dfrac{d}{dr} - \frac{\ell}{r}\right)$
$D_\ell^m = 2\ell(\ell+2)\sqrt{\frac{(\ell+1-m)(\ell+1+m)}{(2\ell+1)(2\ell+3)}} \left(\dfrac{d}{dr}+\frac{\ell+1}{r}\right)$

Note that a common factor of $1/r^2$ has been divided out of each side of the above equations. 

#### Part II buoyancy model:
1) $i \omega \tilde{\mathbf{v}} + 2\hat{\mathbf{z}} \times \tilde{\mathbf{v}} = -\nabla \tilde{P}  - \tilde{N}^2(r) \tilde u_r\hat{\mathbf{r}} + E\nabla^2 \tilde{\mathbf{v}}$

2) $\nabla \cdot \tilde{\mathbf{v}} = 0$

3) $i \omega \tilde{u}_r = \tilde{v}_r$

The added term vanishes under $\hat{r}\cdot \nabla \times$, and:

$$\hat{\mathbf{r}}\cdot\nabla\times\nabla\times (-\tilde N^2\tilde u_r \hat{\mathbf{r}}) = \dfrac{\tilde N^2}{r^2} \Delta_H \tilde u_r$$


In the Toroidal-Poloidal representation:

$$\tilde{\mathbf{v}}\cdot\mathbf{\hat{r}} = \sum_{\ell',m'} \frac{\ell'(\ell'+1)}{r^2} W_{\ell'}^{m'}Y_{\ell'}^{m'}$$

So if we set $\tilde u_r = \sum_{\ell',m'} \frac{\ell'(\ell'+1)}{r^2} U_{\ell'}^{m'}Y_{\ell'}^{m'}$, then $\langle 3.,Y_\ell^m\rangle$ can be written:

$$i\omega U_\ell^m = W_\ell^m$$

And, $\langle \hat{\mathbf{r}} \cdot \nabla \times \nabla \times (\tilde N^2/r^2 \Delta_H \tilde u_r),Y_\ell^m \rangle$:

$$\frac{\ell^2(\ell+1)^2\tilde N^2}{r^4} U_\ell^m$$

Overall the new set of equations is:

1) (unchanged from part I)

$$A_\ell^m Z_\ell^m = C_\ell^m W_{\ell-1}^m + D_\ell^m W_{\ell+1}^m$$

2)

$$A_\ell^m B_\ell^m W_\ell^m = C_\ell^m Z_{\ell-1}^m +D_\ell^m Z_{\ell+1}^m + \frac{\ell^2(\ell+1)^2}{r^2}\tilde N^2(r) U_\ell^m$$

3) 

$$i\omega U_\ell^m = W_\ell^m$$

#### Part III Magnetic field:

1) $i \omega \tilde{\mathbf{v}} + 2\hat{\mathbf{z}} \times \tilde{\mathbf{v}} = -\nabla \tilde{P} + E_\eta B_r \partial_r \tilde{\mathbf{b}} - \tilde{N}^2 \tilde u_r\hat{\mathbf{r}} + E\nabla^2 \tilde{\mathbf{v}}$

2) $\nabla \cdot \tilde{\mathbf{v}} = 0$

3) $i \omega \tilde{u}_r = \tilde{v}_r$
4) $i \omega \tilde{\mathbf{b}} = B_r \partial_r \tilde{\mathbf{v}} + E_\eta \nabla^2 \tilde{\mathbf{b}}$

5) $\nabla \cdot \tilde{\mathbf{b}} = 0$

Introduce a poloidal-toroidal representation for the perturbation magnetic field:

$$\tilde{\mathbf{b}} = \sum_{{\ell'},{m'}}\left(\nabla \times( \nabla \times (S_{\ell'}^{m'}(r) Y_{\ell'}^{m'}(\theta,\varphi)\hat{\mathbf{r}})) + \nabla \times (T_{\ell'}^{m'}(r) Y_{\ell'}^{m'}(\theta,\varphi)\hat{\mathbf{r}})\right)$$

This ensures 5. is satisfied. 

The additional terms are $E_\eta B_r \partial_r\tilde{\mathbf{b}}$, $i\omega \tilde{\mathbf{b}}$, $B_r \partial_r \tilde{\mathbf{v}}$, $E_\eta \nabla^2 \tilde{\mathbf{b}}$.  Note that

$$\hat{\mathbf{r}}\cdot \nabla\times \tilde{\mathbf{v}} = \frac{\Delta_H Z}{r^2}$$
$$\hat{\mathbf{r}}\cdot \nabla\times \tilde{\mathbf{b}} = \frac{\Delta_H T}{r^2}$$
$$\hat{\mathbf{r}}\cdot\nabla\times\nabla\times \tilde{\mathbf{v}}=\dfrac{1}{r^2}\left(\dfrac{\Delta_H}{r^2} - \dfrac{\partial^2}{\partial r^2} \right) \Delta_H W$$
$$\hat{\mathbf{r}}\cdot\nabla\times\nabla\times \tilde{\mathbf{b}}=\dfrac{1}{r^2}\left(\dfrac{\Delta_H}{r^2} - \dfrac{\partial^2}{\partial r^2} \right) \Delta_H S$$

For Lorentz force $\langle \hat{\mathbf{r}}\cdot \nabla \times E_\eta B_r \partial_r \tilde{\mathbf{b}} ,Y_\ell^m\rangle$:

$$E_\eta B_r \dfrac{\ell(\ell+1)}{r^2} \left(\dfrac{d T_\ell^m}{dr} - \dfrac{2T_\ell^m}{r}\right)$$

$\langle \hat{\mathbf{r}} \cdot \nabla \times \nabla \times E_\eta B_r \partial_r \tilde{\mathbf{b}}, Y_\ell^m\rangle$:

$$\dfrac{E_\eta B_r \ell^2(\ell+1)^2}{r^4} \left( \dfrac{d S_\ell^m}{dr} -\dfrac{2 S_\ell^m}{r} \right)$$

For the time derivative term $\langle \hat{\mathbf{r}}\cdot \nabla \times i\omega \tilde{\mathbf{b}} , Y_\ell^m\rangle$:

$$i\omega \dfrac{\ell(\ell+1)}{r^2} T_\ell^m$$

$\langle \hat{\mathbf{r}}\cdot\nabla\times\nabla\times i\omega \tilde{\mathbf{b}},Y_\ell^m \rangle$:

$$\dfrac{i\omega \ell(\ell+1)}{r^2} \left( \dfrac{\ell(\ell+1)}{r^2} - \dfrac{d^2}{dr^2}\right) S_\ell^m$$

For the induction term $\langle \hat{\mathbf{r}} \cdot \nabla \times B_r \partial_r \tilde{\mathbf{v}},Y_\ell^m\rangle$:

$$B_r \dfrac{\ell(\ell+1)}{r^2} \left(\dfrac{dZ_\ell^m}{dr} - \dfrac{2Z_\ell^m}{r}\right)$$

$\langle \hat{\mathbf{r}}\cdot \nabla\times \nabla \times B_r \partial_r \tilde{\mathbf{v}} ,Y_\ell^m\rangle$:

$$B_r \dfrac{\ell^2(\ell+1)^2}{r^4} \left( \frac{d W_\ell^m}{dr} - \dfrac{2 W_\ell^m}{r}\right)$$

For the diffusion term $\langle \hat{\mathbf{r}}\cdot \nabla \times E_\eta \nabla^2 \tilde{\mathbf{b}},Y_\ell^m\rangle$:

$$-E_\eta\frac{\ell(\ell+1)}{r^2}\left(\frac{\ell(\ell+1)}{r^2} - \dfrac{d^2}{dr^2}\right) T_\ell^m$$

$\langle \hat{\mathbf{r}}\cdot \nabla \times\nabla \times E_\eta \nabla^2 \tilde{\mathbf{b}},Y_\ell^m\rangle$:

$$-E_\eta \dfrac{\ell(\ell+1)}{r^2} \left(\dfrac{\ell(\ell+1)}{r^2} - \dfrac{d^2}{dr^2}\right)^2 S_\ell^m$$

Overall the new set of equations is:

1) (unchanged from part I)

$A_\ell^m Z_\ell^m = C_\ell^m W_{\ell-1}^m + D_\ell^m W_{\ell+1}^m+E_\eta B_r\ell(\ell+1) \left(\dfrac{d T_\ell^m}{dr} - \dfrac{2T_\ell^m}{r}\right)$

2)

$A_\ell^m B_\ell^m W_\ell^m = C_\ell^m Z_{\ell-1}^m +D_\ell^m Z_{\ell+1}^m + \dfrac{\ell^2(\ell+1)^2}{r^2}\tilde N^2(r) U_\ell^m + \dfrac{E_\eta B_r \ell^2(\ell+1)^2}{r^2} \left( \dfrac{d S_\ell^m}{dr} -\dfrac{2 S_\ell^m}{r} \right)$

3) 

$i\omega U_\ell^m = W_\ell^m$

4)

$i\omega  T_\ell^m = B_r  \left(\dfrac{dZ_\ell^m}{dr} - \dfrac{2Z_\ell^m}{r}\right) -E_\eta\left(\dfrac{\ell(\ell+1)}{r^2} - \dfrac{d^2}{dr^2}\right) T_\ell^m$

5)

$i\omega \left( \dfrac{\ell(\ell+1)}{r^2} - \dfrac{d^2}{dr^2}\right) S_\ell^m = B_r \dfrac{\ell(\ell+1)}{r^2} \left( \dfrac{d W_\ell^m}{dr} - \dfrac{2 W_\ell^m}{r}\right)-E_\eta  \left(\dfrac{\ell(\ell+1)}{r^2} - \dfrac{d^2}{dr^2}\right)^2 S_\ell^m$

Note a factor of $\dfrac{\ell(\ell+1)}{r^2}$ has been divided out of both sides of 4. and 5.

### Important Notes

1) None of the equations are coupled across spherical harmonic order $m$, so this variable may be taken as a parameter.

2) The momentum equation couples toroidal terms of even (odd) degree $\ell$ with poloidal terms of odd (even) degree. Solutions are separated into symmetry classes $(Z_1,W_2,Z_3,W_4,\dots)^T$ and $(W_1,Z_2,W_3,Z_4,\dots)^T$, symmetric and anti-symmetric about the equator plane. 

3) The displacement and magnetic field equations decouple in both degree and order. 

4) We'll use a combined solution vector, e.g., $(Z_1,T_1,W_2,S_2,D_2,Z_3,T_3,\dots)^T$.



![Description](system.png)




