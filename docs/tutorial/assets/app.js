(function () {
  const root = document.documentElement;
  const saved = localStorage.getItem("quasi-tutorial-theme");
  if (saved) root.setAttribute("data-theme", saved);

  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    const syncLabel = () => {
      const dark = root.getAttribute("data-theme") === "dark";
      themeBtn.textContent = dark ? "Light" : "Dark";
    };
    syncLabel();
    themeBtn.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("quasi-tutorial-theme", next);
      syncLabel();
    });
  }

  document.querySelectorAll(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const pre = btn.closest(".code-card")?.querySelector("pre code");
      if (!pre) return;
      const text = pre.innerText;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
          const orig = btn.textContent;
          btn.textContent = "Copied!";
          setTimeout(() => (btn.textContent = orig), 1400);
        });
      }
    });
  });

  document.querySelectorAll(".tabs").forEach((container) => {
    const btns = container.querySelectorAll(".tab-btn");
    const panels = container.querySelectorAll(".tab-panel");
    btns.forEach((btn, idx) => {
      btn.addEventListener("click", () => {
        btns.forEach((b) => b.classList.remove("active"));
        panels.forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        if (panels[idx]) panels[idx].classList.add("active");
      });
    });
  });

  const wizardData = {
    did: {
      title: "Staggered Difference-in-Differences (PanelDesign / StaggeredDiD)",
      when: "Units adopt treatment at different times (absorbing rollout) and you observe pre- and post-treatment panel outcomes.",
      spec: `spec = quasi.PanelDesign(
    outcome="y", unit="firm", time="year", first_treated="g",
    covariates=("x1",), est_method="dr_improved",
    control_group="not_yet_treated", base_period="universal",
)
res = quasi.estimate(df, design=spec)
honest = res.sensitivity_analysis(method="relative_magnitudes")`,
      link: "did.html",
    },
    rdd: {
      title: "Regression Discontinuity Design (RDDDesign / SharpRDD / FuzzyRDD)",
      when: "Treatment probability jumps sharply (or discontinuously) when a continuous score crosses a known cutoff.",
      spec: `spec = quasi.RDDDesign(
    outcome="gpa", running_var="score", cutoff=75.0,
    bandwidth="mserd", inference="bias_corrected", vce="nn",
)
res = quasi.estimate(df, design=spec)
density = res.cjm_density()`,
      link: "rdd.html",
    },
    iv: {
      title: "Instrumental Variables (IVDesign / TwoStageLS)",
      when: "Treatment is endogenous, but an external instrument Z shifts treatment without directly affecting Y.",
      spec: `spec = quasi.IVDesign(
    outcome="wage", endogenous="schooling", instruments=("proximity", "quarter"),
    covariates=("experience",), covariance="HC1",
)
res = quasi.estimate(df, design=spec)
print(res.effective_f(), res.liml(fuller_alpha=1.0), res.clr_test(0.0))`,
      link: "iv.html",
    },
    its: {
      title: "Interrupted Time Series & Controlled ITS (ITSDesign / ControlledITS)",
      when: "You track a single aggregate time series (or treated + control series) across a sharp policy date.",
      spec: `spec = quasi.ITSDesign(
    outcome="rate", time="month", intervention_date=48,
    error_model="ewc", seasonal_period=12, seasonal_harmonics=2,
)
res = quasi.estimate(df, design=spec)
traj = res.effect_trajectory()`,
      link: "its.html",
    },
    synth: {
      title: "Synthetic Control, SDiD & Augmented SCM (SyntheticDesign / AugmentedSCM)",
      when: "A single (or few) aggregate unit(s) is treated and you have a panel of untreated donor units.",
      spec: `ascm = quasi.AugmentedSCM(
    df, outcome="y", unit="state", time="year",
    treated_unit="California", intervention_time=1989,
).fit()
ci = ascm.conformal_interval(alpha=0.10)`,
      link: "synthetic.html",
    },
    obs: {
      title: "Observational ATT: AIPW, IPW & Matching (ObservationalDesign)",
      when: "Cross-sectional treatment with rich pre-treatment covariates and conditional exchangeability + overlap.",
      spec: `spec = quasi.ObservationalDesign(
    outcome="earnings", treatment="training", covariates=("age", "prior_earn"),
    estimator="aipw", n_splits=5, stabilized=True,
)
res = quasi.estimate(df, design=spec)
ovb = res.ovb_sensitivity()`,
      link: "observational.html",
    },
  };

  const wBtns = document.querySelectorAll(".wizard-btn");
  const wOut = document.getElementById("wizard-output");
  if (wBtns.length && wOut) {
    const renderWizard = (key) => {
      const item = wizardData[key];
      if (!item) return;
      wBtns.forEach((b) => b.classList.toggle("active", b.dataset.key === key));
      wOut.innerHTML =
        `<div style="font-weight:700;font-size:1rem;margin-bottom:4px;">${item.title}</div>` +
        `<p style="margin:0 0 10px;color:var(--muted);font-size:0.9rem;">${item.when}</p>` +
        `<pre style="border-radius:8px;margin-bottom:10px;"><code>${item.spec.replace(/</g, "&lt;")}</code></pre>` +
        `<a href="${item.link}" style="font-weight:700;color:var(--accent);text-decoration:none;">Open full chapter tutorial &rarr;</a>`;
    };
    wBtns.forEach((b) => b.addEventListener("click", () => renderWizard(b.dataset.key)));
    renderWizard("did");
  }
})();
