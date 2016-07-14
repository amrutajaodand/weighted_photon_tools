# weighted_photon_tools

Tools for working with probability-weighted photons.

If appropriately appeased by offerings (for vegetarians we suggest asparagus), the Fermi tools can be made to yield a photon list in which each photon is tagged with the probability of it being from your object of interest. This takes the place of aperture photometry as a way of distinguishing source photons from background, though it should be noted that many of these probabilities are rather small, and unfortunately even much of the total probabilty comes from low-probability photons. It is thus imperative to process these photons with tools that correctly take into account these probability weights.

In the literature you will find that Matthew Kerr has worked out the correct way to compute an H test for such photons. This toolkit includes that, of course, but also includes tools for histogramming and background subtraction. This toolkit is also a work in progress, and will probably acquire new functionality and new interfaces, including probably a Profile object for working with sets of empirical Fourier coefficients (a folded pulse profile in Fourier form).


