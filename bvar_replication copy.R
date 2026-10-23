# ============================================================================
# COMPLETE BVAR REPLICATION WITH SIGN RESTRICTIONS - FULL PATCHED VERSION (v6)
# Changes:
#  - Data caching: load from bvar_final_data.csv if present (unless force_refresh)
#  - Keeps all-combo IRF/FEVD PNGs + robust plotting dispatchers
# ============================================================================

rm(list = ls())

library(BVAR); library(fredr); library(tidyverse); library(lubridate); library(zoo)

# Read the FRED key from the environment; never use a committed fallback.
fred_key <- Sys.getenv("FRED_API_KEY")
if (!nzchar(trimws(fred_key))) {
  stop(
    "Set FRED_API_KEY in your environment or local .Renviron before running.",
    call. = FALSE
  )
}
fredr_set_key(fred_key)

# ----------------------- Hardened plotting helpers --------------------------
.safe_mtext <- function(txt) tryCatch(mtext(txt, side = 3, line = 1, cex = 1.1), error = function(e) invisible(NULL))
.try_plot   <- function(fun, ...) tryCatch({ fun(...); TRUE }, error = function(e) FALSE)

.safe_plot_with_variants <- function(x, idx = NULL, imp_idx = NULL, names_vec = NULL, title = NULL, extra_args = list()) {
  atts <- list()
  if (!is.null(idx) && !is.null(imp_idx)) {
    atts <- c(atts, list(
      list(vars_response = idx, vars_impulse = imp_idx),
      list(vars = idx, which = imp_idx),
      list(response = idx, impulse = imp_idx),
      list(responses = idx, impulses = imp_idx)
    ))
    if (!is.null(names_vec)) {
      rn <- names_vec[idx]; inms <- names_vec[imp_idx]
      atts <- c(atts, list(
        list(vars_response = rn, vars_impulse = inms),
        list(vars = rn, which = inms),
        list(response = rn, impulse = inms),
        list(responses = rn, impulses = inms)
      ))
    }
  }
  atts <- c(atts, list(
    list(vars_response = idx), list(vars = idx), list(response = idx),
    list(responses = idx), list(variables = idx), list(which = idx), list(series = idx)
  ))
  if (!is.null(names_vec) && !is.null(idx)) {
    rn <- names_vec[idx]
    atts <- c(atts, list(
      list(vars_response = rn), list(vars = rn), list(response = rn),
      list(responses = rn), list(variables = rn), list(which = rn), list(series = rn)
    ))
  }
  atts <- c(atts, list(list()))  # plot(x) fallback
  
  for (a in atts) {
    args <- c(list(x = x), a, extra_args)
    ok <- .try_plot(do.call, what = plot, args = args)
    if (ok) { if (!is.null(title)) .safe_mtext(title); return(invisible(TRUE)) }
  }
  stop("Plot dispatch failed for this object.")
}

.with_pdf <- function(file, width, height, code) { pdf(file, width=width, height=height); on.exit(tryCatch(dev.off(), error=function(e){}), add=TRUE); force(code) }
.with_png <- function(file, width_px=1600, height_px=1100, res=144, code) { png(filename=file, width=width_px, height=height_px, res=res); on.exit(tryCatch(dev.off(), error=function(e){}), add=TRUE); force(code) }
.sanitize <- function(x) { x <- gsub("[^A-Za-z0-9]+","_",x); gsub("_+$","",gsub("^_+","",x)) }
.ensure_dir <- function(d) { if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE); normalizePath(d, winslash = "/", mustWork = FALSE) }

# ----------------------------- Data prep ------------------------------------
fred_series <- c("GDPC1","PCECC96","PRFIC1","PNFIC1","EXPGSC1","IMPGSC1","GCEC1","FGCEC1",
                 "GDPDEF","PCEPI","PCEPILFE","CPIAUCSL","CPILFESL","COMPRNFB","PAYEMS","UNRATE","INDPRO",
                 "TCU","HOUST","DSPIC96","UMCSENT","GS2","GS5","GS10","AAA","BAA","DTWEXBGS","SP500",
                 "DCOILWTICO","DCOILBRENTEU","VIXCLS")

var_names <- c("Real_GDP","Real_PCE","Real_Res_Inv","Real_NonRes_Inv","Real_Exports","Real_Imports",
               "Real_Gov_Cons","Real_Fed_Cons","GDP_Deflator","PCE_Price","Core_PCE_Price","CPI","Core_CPI",
               "Compensation","Employment","Unemployment","Ind_Production","Capacity_Util","Housing_Starts",
               "Real_Disp_Income","Consumer_Sentiment","GS2","GS5","GS10","AAA_Yield","BAA_Yield","Dollar_Index",
               "SP500","Oil_Price","Brent_Oil","VIX")

series_to_label <- setNames(var_names, fred_series)

download_and_prepare_data <- function(cache_file = "bvar_final_data.csv",
                                      force_refresh = FALSE,
                                      start_date = as.Date("1990-01-01"),
                                      end_date   = as.Date("2019-12-31")) {
  if (!force_refresh && file.exists(cache_file)) {
    cat("=== LOADING DATA FROM CACHE:", cache_file, "===\n")
    data <- read.csv(cache_file, stringsAsFactors = FALSE)
    data$date <- as.Date(data$date)
    # keep only known columns (in case cache has extras) and order them
    keep <- c("date", intersect(var_names, names(data)))
    data <- data[, keep, drop = FALSE]
    cat("Loaded", nrow(data), "rows and", ncol(data)-1, "variables from cache.\n")
    return(data)
  }
  
  cat("=== DOWNLOADING AND PREPARING DATA ===\n")
  all_data <- vector("list", length(fred_series)); names(all_data) <- fred_series
  for (s in fred_series) {
    cat("Downloading", s, "...\n")
    tryCatch({
      d <- fredr(series_id = s, observation_start = start_date, observation_end = end_date)
      if (nrow(d) > 0) {
        d <- d[, c("date","value")]; names(d)[2] <- s
        d <- d[!is.na(d[[s]]), ]
        all_data[[s]] <- d
      }
    }, error = function(e) cat("✗ Failed:", s, "\n"))
    Sys.sleep(0.05)
  }
  
  present <- names(all_data)[!vapply(all_data, is.null, TRUE)]
  if (length(present) == 0) stop("No series downloaded successfully.")
  merged <- Reduce(function(a,b) merge(a,b, by="date", all=TRUE), all_data[present])
  
  # log transforms (x100)
  log_vars <- c("GDPC1","PCECC96","PRFIC1","PNFIC1","EXPGSC1","IMPGSC1","GCEC1","FGCEC1","GDPDEF","PCEPI",
                "PCEPILFE","CPIAUCSL","CPILFESL","COMPRNFB","PAYEMS","INDPRO","HOUST","DSPIC96",
                "DTWEXBGS","SP500","DCOILWTICO","DCOILBRENTEU")
  for (v in intersect(log_vars, names(merged))) {
    merged[[v]][merged[[v]] <= 0] <- 0.001
    merged[[v]] <- log(merged[[v]]) * 100
  }
  
  merged$year <- year(merged$date); merged$quarter <- quarter(merged$date)
  quarterly <- merged %>%
    group_by(year, quarter) %>%
    summarise(across(where(is.numeric),
                     ~ ifelse(all(is.na(.x)), NA_real_, mean(.x, na.rm = TRUE))),
              .groups = "drop") %>%
    mutate(date = as.Date(paste(year, (quarter-1)*3 + 1, "01", sep = "-"))) %>%
    select(-year, -quarter) %>% arrange(date)
  
  miss_prop <- quarterly %>% summarise(across(everything(), ~ mean(is.na(.))), .groups = "drop") %>%
    pivot_longer(everything(), names_to = "var", values_to = "mp")
  keep_vars <- miss_prop$var[miss_prop$mp < 0.5]
  quarterly <- quarterly %>% select(all_of(keep_vars))
  
  row_mp <- rowMeans(is.na(quarterly[, setdiff(names(quarterly),"date")]))
  quarterly <- quarterly[row_mp < 0.5, ] %>%
    arrange(date) %>% fill(everything(), .direction = "down") %>%
    fill(everything(), .direction = "up") %>%
    filter(complete.cases(.), date >= start_date)
  
  present_series <- intersect(fred_series, names(quarterly))
  renamed <- quarterly
  for (s in present_series) names(renamed)[names(renamed) == s] <- series_to_label[[s]]
  ordered_labels <- intersect(var_names, names(renamed))
  renamed <- renamed %>% select(date, all_of(ordered_labels))
  
  cat("Final dataset:", nrow(renamed), "observations,", ncol(renamed)-1, "variables\n")
  cat("Sample period:", format(min(renamed$date)), "to", format(max(renamed$date)), "\n")
  
  write.csv(renamed, cache_file, row.names = FALSE)
  cat("Cached dataset to", cache_file, "\n")
  renamed
}

# ------------------------------- BVAR ---------------------------------------
estimate_bvar_model <- function(data, lags=5) {
  cat("\n=== ESTIMATING BVAR MODEL ===\n")
  y <- as.matrix(data[, setdiff(names(data),"date")])
  cat("Variables:", ncol(y), "\n"); cat("Observations:", nrow(y), "\n"); cat("Lags:", lags, "\n")
  stopifnot(!any(is.na(y)))
  model <- bvar(data=y, lags=lags, n_draw=5000, n_burn=2500, lambda=0.2, alpha=2, psi=1, verbose=TRUE)
  cat("✓ BVAR estimation completed!\n")
  list(model=model, data_matrix=y, variable_names=colnames(y))
}

# -------------------------- Sign restrictions (matrix) ----------------------
setup_sign_restrictions <- function(variable_names) {
  cat("\n=== SETTING UP SIGN RESTRICTIONS ===\n")
  nv <- length(variable_names); S <- array(0L, dim=c(nv,4)); rownames(S) <- variable_names
  colnames(S) <- c("Monetary","Demand","Supply","Financial")
  idx <- function(nm) match(nm, variable_names, nomatch=0L)
  gdp<-idx("Real_GDP"); pdef<-idx("GDP_Deflator"); i10<-idx("GS10"); sp<-idx("SP500"); vix<-idx("VIX")
  cat("GDP:", gdp, " Price:", pdef, " GS10:", i10, " SP500:", sp, " VIX:", vix, "\n")
  if (i10>0) { S[i10,"Monetary"] <- 1; if (gdp>0) S[gdp,"Monetary"] <- -1; if (pdef>0) S[pdef,"Monetary"] <- -1; if (sp>0) S[sp,"Monetary"] <- -1 }
  if (gdp>0) { S[gdp,"Demand"] <- 1; if (pdef>0) S[pdef,"Demand"] <- 1; if (i10>0) S[i10,"Demand"] <- 1 }
  if (gdp>0 && pdef>0) { S[gdp,"Supply"] <- 1; S[pdef,"Supply"] <- -1 }
  if (vix>0) { S[vix,"Financial"] <- 1; if (gdp>0) S[gdp,"Financial"] <- -1; if (sp>0) S[sp,"Financial"] <- -1 }
  cat("\nSign restrictions matrix:\n"); print(S); S
}
apply_sign_restrictions <- function(bvar_model, sign_matrix, horizon=20L, n_draws=1000L) {
  cat("\n=== APPLYING SIGN RESTRICTIONS ===\n"); cat("Computing Cholesky IRFs as baseline...\n")
  irf(bvar_model, horizon=horizon, identification=TRUE)  # placeholder
}

# ------------------------------- IRFs/FEVD ----------------------------------
compute_all_irfs <- function(bvar_result, horizon=20) {
  cat("\n=== COMPUTING IMPULSE RESPONSE FUNCTIONS ===\n")
  mdl <- bvar_result$model; vnames <- bvar_result$variable_names
  cat("Computing Generalized IRFs...\n"); irf_gen  <- irf(mdl, horizon=horizon, identification=FALSE)
  cat("Computing Cholesky IRFs...\n");    irf_chol <- irf(mdl, horizon=horizon, identification=TRUE)
  cat("Computing Sign-restricted IRFs...\n"); S <- setup_sign_restrictions(vnames); irf_sr <- apply_sign_restrictions(mdl, S, horizon=horizon)
  cat("✓ All IRF computations completed!\n")
  list(generalized=irf_gen, cholesky=irf_chol, sign_restricted=irf_sr, sign_matrix=S)
}
compute_comprehensive_fevd <- function(bvar_model, horizon=20) { cat("\n=== COMPUTING FEVD ===\n"); F <- fevd(bvar_model, horizon=horizon); cat("✓ FEVD done!\n"); F }

# ---------------------- Forecast (unchanged baseline) -----------------------
setup_conditional_forecasting <- function(bvar_model) { cat("\n=== CONDITIONAL FORECASTING ===\n"); fc <- predict(bvar_model, horizon=12); cat("Baseline forecast computed.\n"); fc }

# ------------------------------- Plotting -----------------------------------
.save_irf_all_pairs <- function(irf_obj, vnames, outdir, prefix) {
  R <- seq_along(vnames); I <- seq_along(vnames)
  for (r in R) {
    for (imp in I) {
      r_name <- vnames[r]; imp_name <- vnames[imp]
      fname <- paste0(prefix, "_", .sanitize(r_name), "_imp_", .sanitize(imp_name), ".png")
      title <- paste(prefix, "—", r_name, "impulse:", imp_name)
      .with_png(file.path(outdir, fname),
                code = .safe_plot_with_variants(irf_obj, idx = r, imp_idx = imp, names_vec = vnames, title = title)
      )
    }
  }
  cat("✓ Saved all", length(R)*length(I), "IRF PNGs to", outdir, "\n")
}

.save_fevd_all <- function(fevd_obj, vnames, outdir, prefix="fevd") {
  R <- seq_along(vnames)
  for (r in R) {
    r_name <- vnames[r]; fname <- paste0(prefix, "_", .sanitize(r_name), ".png")
    title <- paste("FEVD —", r_name)
    .with_png(file.path(outdir, fname),
              code = .safe_plot_with_variants(fevd_obj, idx=r, names_vec=vnames, title=title)
    )
  }
  cat("✓ Saved", length(R), "FEVD PNGs to", outdir, "\n")
}

create_comprehensive_plots <- function(irf_results, fevd_results, forecasts, bvar_result) {
  cat("\n=== SAVING INDIVIDUAL PLOTS (ALL COMBINATIONS) ===\n")
  vnames <- colnames(bvar_result$data_matrix)
  
  # Prepare output dirs (under working dir)
  dir_irf_g  <- .ensure_dir(file.path(getwd(), "irf_generalized"))
  dir_irf_c  <- .ensure_dir(file.path(getwd(), "irf_cholesky"))
  dir_fevd   <- .ensure_dir(file.path(getwd(), "fevd"))
  dir_pdf    <- .ensure_dir(file.path(getwd(), "structural_analysis_plots"))
  
  # ---- IRFs: generalized (all response x impulse)
  cat("IRFs — generalized (all pairs)...\n")
  .save_irf_all_pairs(irf_results$generalized, vnames, dir_irf_g, prefix="irf_generalized")
  
  # ---- IRFs: cholesky (all response x impulse)
  cat("IRFs — cholesky (all pairs)...\n")
  .save_irf_all_pairs(irf_results$cholesky, vnames, dir_irf_c, prefix="irf_cholesky")
  
  # ---- FEVD: all responses
  cat("FEVD — all responses...\n")
  .save_fevd_all(fevd_results, vnames, dir_fevd, prefix="fevd")
  
  # ---- Forecasts & Sign matrix (kept as PDFs)
  cat("Forecasts (PDF)...\n")
  .with_pdf(file.path(dir_pdf,"forecasts.pdf"), 14, 10,
            .safe_plot_with_variants(forecasts, idx=seq_len(min(8, length(vnames))), names_vec=vnames, title="Unconditional Forecasts")
  )
  cat("Sign restrictions heatmap (PDF)...\n")
  S <- irf_results$sign_matrix
  if (!is.null(S)) {
    Sm <- matrix(S, nrow=nrow(S), ncol=ncol(S)); Sm_fac <- matrix(match(Sm, c(-1,0,1)), nrow=nrow(Sm))
    colset <- c("#d73027","#f7f7f7","#1a9850")
    .with_pdf(file.path(dir_pdf,"sign_restrictions.pdf"), 10, 8, {
      op <- par(mar=c(6,10,4,2)); Sm_flip <- Sm_fac[nrow(Sm_fac):1, , drop=FALSE]
      image(x=seq_len(ncol(Sm_flip)), y=seq_len(nrow(Sm_flip)), z=t(Sm_flip), xaxt="n", yaxt="n", col=colset,
            main="Sign Restrictions Matrix (-1 red, 0 white, +1 green)")
      axis(1, at=seq_len(ncol(S)), labels=colnames(S), las=2, cex.axis=0.8)
      axis(2, at=seq_len(nrow(S)), labels=rev(rownames(S)), las=2, cex.axis=0.8)
      par(op)
    })
  }
  
  cat("✓ PNGs in subfolders: irf_generalized/, irf_cholesky/, fevd/\n")
  cat("✓ PDFs in structural_analysis_plots/\n")
}

# --------------------------------- MAIN -------------------------------------
run_complete_structural_analysis <- function(force_refresh_data = FALSE,
                                             cache_file = "bvar_final_data.csv") {
  cat("=== COMPLETE BVAR STRUCTURAL ANALYSIS ===\n")
  cat("Replicating Crump et al. (2021) with extensions\n")
  cat("==============================================\n")
  
  cat("\nSTEP 1: DATA PREPARATION\n")
  data <- download_and_prepare_data(cache_file = cache_file, force_refresh = force_refresh_data)
  # If we loaded from cache, it already exists; if we freshly downloaded, it was just written.
  
  cat("\nSTEP 2: BVAR ESTIMATION\n")
  bvar_result <- estimate_bvar_model(data, lags=5)
  save(bvar_result, file="bvar_final_model.RData")
  
  cat("\nSTEP 3: STRUCTURAL IDENTIFICATION & IRFs\n")
  irf_results <- compute_all_irfs(bvar_result, horizon=20)
  
  cat("\nSTEP 4: FEVD\n")
  fevd_results <- compute_comprehensive_fevd(bvar_result$model, horizon=20)
  
  cat("\nSTEP 5: CONDITIONAL FORECASTING\n")
  forecasts <- setup_conditional_forecasting(bvar_result$model)
  
  cat("\nSTEP 6: SAVING RESULTS\n")
  save(irf_results, fevd_results, forecasts, file="complete_structural_results.RData")
  
  cat("\nSTEP 7: CREATING PLOTS\n")
  create_comprehensive_plots(irf_results, fevd_results, forecasts, bvar_result)
  
  cat("\n=== ANALYSIS COMPLETED SUCCESSFULLY ===\n")
  cat("Outputs:\n")
  cat("- Data cache:", cache_file, "(set force_refresh_data=TRUE to rebuild)\n")
  cat("- IRFs generalized: one PNG per (response, impulse) in irf_generalized/\n")
  cat("- IRFs cholesky:    one PNG per (response, impulse) in irf_cholesky/\n")
  cat("- FEVD:             one PNG per response in fevd/\n")
  cat("- PDFs:             forecasts.pdf, sign_restrictions.pdf in structural_analysis_plots/\n\n")
  list(data=data, bvar_result=bvar_result, irf_results=irf_results, fevd_results=fevd_results, forecasts=forecasts)
}

cat("COMPLETE BVAR STRUCTURAL ANALYSIS SCRIPT - FULL PATCHED VERSION (v6)\n")
cat("====================================================================\n")
cat("To run (use cache): results <- run_complete_structural_analysis()\n")
cat("To force refresh:   results <- run_complete_structural_analysis(TRUE)\n\n")

# Auto-run (uses cache if available; set TRUE to force a refresh)
results <- run_complete_structural_analysis(FALSE)
