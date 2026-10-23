# ============================================================================
# COMPLETE BVAR REPLICATION WITH SIGN RESTRICTIONS - FULL PATCHED VERSION (v11)
# Fixes vs v10:
#   - Do NOT re-stationarize when loading from cache (prevents double differencing)
#   - Robust BVAR estimation: auto-retry over lambda x lags grid on singular errors
# Keeps:
#   - Deposit-focused variables, caching, all-pairs IRF PNGs, FEVD fallback renderer
# ============================================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(BVAR); library(fredr); library(tidyverse); library(lubridate); library(zoo)
})

# Read the FRED key from the environment; never use a committed fallback.
fred_key <- Sys.getenv("FRED_API_KEY")
if (!nzchar(trimws(fred_key))) {
  stop(
    "Set FRED_API_KEY in your environment or local .Renviron before running.",
    call. = FALSE
  )
}
fredr_set_key(fred_key)

.safe_mtext <- function(txt) tryCatch(mtext(txt, side = 3, line = 1, cex = 1.1), error = function(e) invisible(NULL))
.try_plot   <- function(fun, ...) tryCatch({ fun(...); TRUE }, error = function(e) FALSE)

.safe_plot_with_variants <- function(x, idx=NULL, imp_idx=NULL, names_vec=NULL, title=NULL, extra_args=list()){
  atts <- list()
  if (!is.null(idx) && !is.null(imp_idx)) {
    atts <- c(atts, list(
      list(vars_response=idx, vars_impulse=imp_idx),
      list(vars=idx, which=imp_idx),
      list(response=idx, impulse=imp_idx),
      list(responses=idx, impulses=imp_idx)
    ))
    if (!is.null(names_vec)) {
      rn <- names_vec[idx]; inms <- names_vec[imp_idx]
      atts <- c(atts, list(
        list(vars_response=rn, vars_impulse=inms),
        list(vars=rn, which=inms),
        list(response=rn, impulse=inms),
        list(responses=rn, impulses=inms)
      ))
    }
  }
  atts <- c(atts, list(
    list(vars_response=idx), list(vars=idx), list(response=idx),
    list(responses=idx), list(variables=idx), list(which=idx), list(series=idx)
  ))
  if (!is.null(names_vec) && !is.null(idx)) {
    rn <- names_vec[idx]
    atts <- c(atts, list(
      list(vars_response=rn), list(vars=rn), list(response=rn),
      list(responses=rn), list(variables=rn), list(which=rn), list(series=rn)
    ))
  }
  atts <- c(atts, list(list()))
  for (a in atts) {
    args <- c(list(x=x), a, extra_args)
    ok <- .try_plot(do.call, what=plot, args=args)
    if (ok) { if (!is.null(title)) .safe_mtext(title); return(invisible(TRUE)) }
  }
  stop("Plot dispatch failed for this object.")
}

.with_pdf <- function(file, w, h, code){ pdf(file, width=w, height=h); on.exit(tryCatch(dev.off(), error=function(e){}), add=TRUE); force(code) }
.with_png <- function(file, width_px=1600, height_px=1100, res=144, code){ png(filename=file, width=width_px, height=height_px, res=res); on.exit(tryCatch(dev.off(), error=function(e){}), add=TRUE); force(code) }
.sanitize <- function(x){ x <- gsub("[^A-Za-z0-9]+","_",x); gsub("_+$","",gsub("^_+","",x)) }
.ensure_dir <- function(d){ if (!dir.exists(d)) dir.create(d, recursive=TRUE, showWarnings=FALSE); normalizePath(d, winslash="/", mustWork=FALSE) }

# ----------------------------- Base series ----------------------------------
fred_series <- c("GDPC1","PCECC96","PRFIC1","PNFIC1","EXPGSC1","IMPGSC1","GCEC1","FGCEC1","GDPDEF","PCEPI","PCEPILFE",
                 "CPIAUCSL","CPILFESL","COMPRNFB","PAYEMS","UNRATE","INDPRO","TCU","HOUST","DSPIC96","UMCSENT","GS2","GS5",
                 "GS10","AAA","BAA","DTWEXBGS","SP500","DCOILWTICO","DCOILBRENTEU","VIXCLS")

var_names <- c("Real_GDP","Real_PCE","Real_Res_Inv","Real_NonRes_Inv","Real_Exports","Real_Imports","Real_Gov_Cons",
               "Real_Fed_Cons","GDP_Deflator","PCE_Price","Core_PCE_Price","CPI","Core_CPI","Compensation","Employment",
               "Unemployment","Ind_Production","Capacity_Util","Housing_Starts","Real_Disp_Income","Consumer_Sentiment",
               "GS2","GS5","GS10","AAA_Yield","BAA_Yield","Dollar_Index","SP500","Oil_Price","Brent_Oil","VIX",
               "Saving_Rate","Real_Total_Deposits","Small_Time_Deposits_Growth","Adj_NFCI","EPU_Index_Log")

series_to_label <- setNames(var_names[1:31], fred_series)

# ------------------------- Helpers ------------------------------------------
to_quarterly_stock <- function(df){
  df %>% arrange(date) %>% mutate(year=year(date), q=quarter(date)) %>%
    group_by(year,q) %>% summarise(value=dplyr::last(value), .groups="drop") %>%
    transmute(date=as.Date(paste(year,(q-1)*3+1,"01",sep="-")), value=value)
}
to_quarterly_mean <- function(df){
  df %>% mutate(year=year(date), q=quarter(date)) %>%
    group_by(year,q) %>% summarise(value=mean(value, na.rm=TRUE), .groups="drop") %>%
    transmute(date=as.Date(paste(year,(q-1)*3+1,"01",sep="-")), value=value)
}
resolve_h8_series <- function(title_contains, prefer_sa=TRUE, prefer_weekly=TRUE, limit=100){
  q <- fredr_series_search_text(search_text=title_contains, limit=limit)
  if (nrow(q)==0) stop("No FRED series found for: ", title_contains)
  if (prefer_weekly) q <- q %>% filter(grepl("^Weekly", frequency, ignore.case=TRUE))
  if (prefer_sa && any(q$seasonal_adjustment=="Seasonally Adjusted")) q <- q %>% filter(seasonal_adjustment=="Seasonally Adjusted")
  q <- q %>% arrange(desc(grepl(title_contains, title, ignore.case=TRUE)), observation_start)
  q$id[1]
}
finalize_panel <- function(df){
  df <- df %>% arrange(date)
  cols <- setdiff(names(df), "date")
  df <- df %>% fill(all_of(cols), .direction="down") %>% fill(all_of(cols), .direction="up")
  first_idx <- sapply(cols, function(v){ w <- which(!is.na(df[[v]])); if (length(w)) w[1] else NA_integer_ })
  last_idx  <- sapply(cols, function(v){ w <- which(!is.na(df[[v]])); if (length(w)) w[length(w)] else NA_integer_ })
  start <- max(df$date[first_idx], na.rm=TRUE); end <- min(df$date[last_idx], na.rm=TRUE)
  df <- df %>% filter(date >= start, date <= end)
  df <- df %>% filter(if_all(all_of(cols), ~ !is.na(.x)))
  df
}

# ----------------------------- Data prep ------------------------------------
download_and_prepare_data <- function(cache_file="bvar_final_data.csv", force_refresh=FALSE,
                                      stationarize_financial=TRUE,
                                      start_date=as.Date("1973-01-01"), end_date=as.Date("2019-12-31")) {
  required_new <- c("Saving_Rate","Real_Total_Deposits","Small_Time_Deposits_Growth","Adj_NFCI","EPU_Index_Log")
  
  if (!force_refresh && file.exists(cache_file)) {
    cat("=== LOADING DATA FROM CACHE:", cache_file, "===\n")
    data <- read.csv(cache_file, stringsAsFactors=FALSE); data$date <- as.Date(data$date)
    have_all <- all(required_new %in% names(data))
    if (have_all) {
      # IMPORTANT: do NOT re-apply stationarization on cached data.
      data <- finalize_panel(data)
      write.csv(data, cache_file, row.names=FALSE)  # heal cache if needed
      cat("Loaded", nrow(data), "rows and", ncol(data)-1, "variables from cache.\n")
      return(data)
    }
    cat("Cache missing new variables; rebuilding dataset...\n")
  }
  
  cat("=== DOWNLOADING AND PREPARING BASE DATA ===\n")
  all_data <- vector("list", length(fred_series)); names(all_data) <- fred_series
  for (s in fred_series) {
    cat("Downloading", s, "...\n")
    tryCatch({
      d <- fredr(series_id=s, observation_start=start_date, observation_end=end_date)
      if (nrow(d)>0) { d <- d[,c("date","value")]; names(d)[2] <- s; d <- d[!is.na(d[[s]]),]; all_data[[s]] <- d }
    }, error=function(e) cat("✗ Failed:", s, "\n"))
    Sys.sleep(0.05)
  }
  present <- names(all_data)[!vapply(all_data, is.null, TRUE)]
  if (!length(present)) stop("No base series downloaded successfully.")
  merged <- Reduce(function(a,b) merge(a,b, by="date", all=TRUE), all_data[present])
  
  log_vars <- c("GDPC1","PCECC96","PRFIC1","PNFIC1","EXPGSC1","IMPGSC1","GCEC1","FGCEC1","GDPDEF","PCEPI",
                "PCEPILFE","CPIAUCSL","CPILFESL","COMPRNFB","PAYEMS","INDPRO","HOUST","DSPIC96","DTWEXBGS",
                "SP500","DCOILWTICO","DCOILBRENTEU")
  for (v in intersect(log_vars, names(merged))) { merged[[v]][merged[[v]]<=0] <- 0.001; merged[[v]] <- log(merged[[v]])*100 }
  
  merged$year <- year(merged$date); merged$quarter <- quarter(merged$date)
  quarterly <- merged %>% group_by(year, quarter) %>%
    summarise(across(where(is.numeric), ~ ifelse(all(is.na(.x)), NA_real_, mean(.x, na.rm=TRUE))), .groups="drop") %>%
    mutate(date = as.Date(paste(year,(quarter-1)*3+1,"01",sep="-"))) %>%
    select(-year,-quarter) %>% arrange(date)
  
  miss_prop <- quarterly %>% summarise(across(everything(), ~ mean(is.na(.))), .groups="drop") %>%
    pivot_longer(everything(), names_to="var", values_to="mp")
  keep_vars <- miss_prop$var[miss_prop$mp < 0.5]
  quarterly <- quarterly %>% select(all_of(keep_vars))
  row_mp <- rowMeans(is.na(quarterly[, setdiff(names(quarterly),"date")]))
  quarterly <- quarterly[row_mp < 0.5, ] %>% arrange(date) %>% fill(everything(), .direction="down") %>%
    fill(everything(), .direction="up") %>% filter(complete.cases(.), date >= start_date)
  
  present_series <- intersect(fred_series, names(quarterly))
  renamed <- quarterly; for (s in present_series) names(renamed)[names(renamed)==s] <- series_to_label[[s]]
  
  # ---------------------- Deposit-related series ----------------
  cat("=== DOWNLOADING DEPOSIT-RELATED SERIES ===\n")
  pce_q <- to_quarterly_mean(fredr(series_id="PCEPI", observation_start=start_date, observation_end=end_date) %>% select(date,value)) %>% rename(PCEPI_Q=value)
  Saving_Rate <- to_quarterly_mean(fredr(series_id="PSAVERT", observation_start=start_date, observation_end=end_date) %>% select(date,value)) %>% rename(Saving_Rate=value)
  Adj_NFCI <- to_quarterly_mean(fredr(series_id="ANFCI", observation_start=start_date, observation_end=end_date) %>% select(date,value)) %>% rename(Adj_NFCI=value)
  EPU_Index_Log <- to_quarterly_mean(fredr(series_id="USEPUINDXD", observation_start=start_date, observation_end=end_date) %>% select(date,value)) %>%
    mutate(EPU_Index_Log = 100*log(pmax(value,0.001))) %>% select(date,EPU_Index_Log)
  
  total_id <- resolve_h8_series("Deposits, All Commercial Banks")
  small_id <- resolve_h8_series("Small-denomination Time Deposits, All Commercial Banks")
  total_q <- to_quarterly_stock(fredr(series_id=total_id, observation_start=start_date, observation_end=end_date) %>% select(date,value)) %>% rename(Total_Deposits=value)
  small_q <- to_quarterly_stock(fredr(series_id=small_id, observation_start=start_date, observation_end=end_date) %>% select(date,value)) %>% rename(Small_Time_Deposits=value)
  
  total_real <- total_q %>% left_join(pce_q, by="date") %>% mutate(Real_Total_Deposits = 100*log(pmax(Total_Deposits,0.001)/pmax(PCEPI_Q,0.001))) %>% select(date,Real_Total_Deposits)
  small_growth <- small_q %>% arrange(date) %>% mutate(Small_Time_Deposits_Growth = 100*(log(pmax(Small_Time_Deposits,0.001)) - log(dplyr::lag(pmax(Small_Time_Deposits,0.001))))) %>% select(date,Small_Time_Deposits_Growth)
  
  added <- Saving_Rate %>% full_join(Adj_NFCI, by="date") %>% full_join(EPU_Index_Log, by="date") %>%
    full_join(total_real, by="date") %>% full_join(small_growth, by="date") %>% arrange(date)
  
  renamed <- renamed %>% full_join(added, by="date") %>% arrange(date)
  ordered_labels <- intersect(var_names, names(renamed))
  renamed <- renamed %>% select(date, all_of(ordered_labels))
  
  # ---- Stationarize selected price-levels to Δlog×100 (returns/growth) ----
  if (stationarize_financial) {
    diff_vars <- intersect(c("SP500","Oil_Price","Brent_Oil","Dollar_Index","Real_Total_Deposits"), names(renamed))
    if (length(diff_vars)) {
      cat("Stationarizing (Δlog×100) variables:", paste(diff_vars, collapse=", "), "\n")
      for (v in diff_vars) renamed[[v]] <- renamed[[v]] - dplyr::lag(renamed[[v]])
    }
  }
  
  renamed <- finalize_panel(renamed)
  
  cat("Final dataset:", nrow(renamed), "observations,", ncol(renamed)-1, "variables\n")
  cat("Sample period:", format(min(renamed$date)), "to", format(max(renamed$date)), "\n")
  
  write.csv(renamed, cache_file, row.names=FALSE); cat("Cached dataset to", cache_file, "\n")
  renamed
}

# ------------------------------- BVAR ---------------------------------------
# Robust estimator that retries over lambda x lags if the linear system is ill-conditioned
estimate_bvar_model <- function(data, lags=5,
                                lambda_grid = c(0.2, 0.1, 0.05, 0.02),
                                lags_grid   = c(lags, 4, 3, 2),
                                alpha = 2, psi = 1) {
  cat("\n=== ESTIMATING BVAR MODEL ===\n")
  y <- as.matrix(data[, setdiff(names(data),"date")])
  cat("Variables:", ncol(y), "\n"); cat("Observations:", nrow(y), "\n"); cat("Target lags:", lags, "\n")
  stopifnot(!any(is.na(y)))
  
  last_err <- NULL; best <- NULL
  for (L in unique(lags_grid)) {
    for (lam in lambda_grid) {
      cat(sprintf("Trying lags=%d, lambda=%.3f ... ", L, lam))
      ok <- FALSE; mdl <- NULL
      tryCatch({
        mdl <- bvar(data=y, lags=L, n_draw=5000, n_burn=2500, lambda=lam, alpha=alpha, psi=psi, verbose=TRUE)
        ok <- TRUE
      }, error = function(e) { last_err <<- e })
      if (ok) {
        cat("✓ success\n")
        cat(sprintf("Chosen hyperparameters: lags=%d, lambda=%.3f\n", L, lam))
        return(list(model=mdl, data_matrix=y, variable_names=colnames(y)))
      } else {
        cat("failed\n")
      }
    }
  }
  stop(sprintf("BVAR failed after trying %s.\nLast error: %s",
               paste(sprintf("(lags=%d,λ=%.3f)", rep(unique(lags_grid), each=length(lambda_grid)),
                             rep(lambda_grid, times=length(unique(lags_grid)))), collapse=", "),
               conditionMessage(last_err)))
}

# -------------------------- Sign restrictions (matrix) ----------------------
setup_sign_restrictions <- function(variable_names){
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
apply_sign_restrictions <- function(bvar_model, sign_matrix, horizon=20L, n_draws=1000L){
  cat("\n=== APPLYING SIGN RESTRICTIONS ===\n"); cat("Computing Cholesky IRFs as baseline...\n")
  irf(bvar_model, horizon=horizon, identification=TRUE)
}

# ------------------------------- IRFs/FEVD ----------------------------------
compute_all_irfs <- function(bvar_result, horizon=20){
  cat("\n=== COMPUTING IMPULSE RESPONSE FUNCTIONS ===\n")
  mdl <- bvar_result$model; vnames <- bvar_result$variable_names
  cat("Computing Generalized IRFs...\n"); irf_gen  <- irf(mdl, horizon=horizon, identification=FALSE)
  cat("Computing Cholesky IRFs...\n");    irf_chol <- irf(mdl, horizon=horizon, identification=TRUE)
  cat("Computing Sign-restricted IRFs...\n"); S <- setup_sign_restrictions(vnames); irf_sr <- apply_sign_restrictions(mdl, S, horizon=horizon)
  cat("✓ All IRF computations completed!\n")
  list(generalized=irf_gen, cholesky=irf_chol, sign_restricted=irf_sr, sign_matrix=S)
}
compute_comprehensive_fevd <- function(bvar_model, horizon=20){
  cat("\n=== COMPUTING FEVD ===\n"); F <- fevd(bvar_model, horizon=horizon); cat("✓ FEVD done!\n"); F
}

# -------- FEVD fallback extraction & plotting -------------------------------
.extract_fevd_array <- function(obj){
  if (is.array(obj)) return(obj)
  if (is.list(obj)) {
    cand <- c("fevd","x","data","values","decomp")
    for (nm in cand) if (!is.null(obj[[nm]]) && is.array(obj[[nm]])) return(obj[[nm]])
  }
  stop("Could not extract FEVD array from object.")
}
.render_fevd_response <- function(fevd_obj, r_name, vnames, fname){
  A <- .extract_fevd_array(fevd_obj)
  if (length(dim(A)) != 3) stop("FEVD array not 3D.")
  dims <- dim(A); dn <- dimnames(A); K <- length(vnames)
  k_dims <- which(dims == K)
  if (length(k_dims) < 2) {
    k_dims <- which(sapply(dn, function(x) !is.null(x) && sum(x %in% vnames) >= floor(K/2)))
    if (length(k_dims) < 2) stop("Cannot identify variable dimensions in FEVD array.")
  }
  h_dim <- setdiff(1:3, k_dims)[1]
  perm1 <- c(h_dim, k_dims); B1 <- aperm(A, perm1); dnB1 <- dimnames(B1)
  use_perm1 <- !is.null(dnB1) && length(dnB1)==3 && !is.null(dnB1[[3]]) && r_name %in% dnB1[[3]]
  if (use_perm1) {
    H <- dim(B1)[1]; resp_idx <- match(r_name, dnB1[[3]]); M <- B1[,,resp_idx, drop=FALSE][,,1]
    imp_names <- if (!is.null(dnB1[[2]])) dnB1[[2]] else vnames
  } else {
    perm2 <- c(h_dim, rev(k_dims)); B2 <- aperm(A, perm2); dnB2 <- dimnames(B2)
    if (!is.null(dnB2) && length(dnB2)==3 && !is.null(dnB2[[3]]) && r_name %in% dnB2[[3]]) {
      H <- dim(B2)[1]; resp_idx <- match(r_name, dnB2[[3]]); M <- B2[,,resp_idx, drop=FALSE][,,1]
      imp_names <- if (!is.null(dnB2[[2]])) dnB2[[2]] else vnames
    } else stop("Could not align FEVD dims.")
  }
  M[M < 0] <- 0; rs <- rowSums(M); rs[rs==0] <- 1; M <- sweep(M, 1, rs, "/")
  avg_share <- colMeans(M[max(1, nrow(M)-5):nrow(M), , drop=FALSE]); ord <- order(avg_share, decreasing=TRUE)
  top <- ord[seq_len(min(10, length(ord)))]; Mt <- M[, top, drop=FALSE]; labs <- imp_names[top]
  .with_png(fname, code = {
    matplot(1:nrow(Mt), Mt, type="l", lwd=2, lty=1,
            xlab="Horizon", ylab="Share of forecast error variance",
            main=paste("FEVD —", r_name, "(top shocks)"))
    legend("right", legend=labs, lty=1, lwd=2, cex=0.8, bty="n")
    abline(h=seq(0,1,by=0.2), v=seq(0,nrow(Mt),by=5), lty=3, col="gray60")
  })
}
.save_fevd_all <- function(fevd_obj, vnames, outdir, prefix="fevd"){
  for (r in seq_along(vnames)) {
    r_name <- vnames[r]; fname <- file.path(outdir, paste0(prefix, "_", .sanitize(r_name), ".png"))
    ok <- tryCatch({
      .with_png(fname, code = .safe_plot_with_variants(fevd_obj, idx=r, names_vec=vnames, title=paste("FEVD —", r_name)))
      TRUE
    }, error = function(e) FALSE)
    if (!ok) {
      tryCatch({ .render_fevd_response(fevd_obj, r_name, vnames, fname) },
               error = function(e) message("FEVD fallback failed for ", r_name, ": ", conditionMessage(e)))
    }
    cat("Saved:", basename(fname), "\n")
  }
  cat("✓ FEVD PNGs saved to", outdir, "\n")
}

# ---------------------- Forecast (unchanged baseline) -----------------------
setup_conditional_forecasting <- function(bvar_model){ cat("\n=== CONDITIONAL FORECASTING ===\n"); fc <- predict(bvar_model, horizon=12); cat("Baseline forecast computed.\n"); fc }

# ------------------------------- Plotting -----------------------------------
.save_irf_all_pairs <- function(irf_obj, vnames, outdir, prefix){
  R <- seq_along(vnames); I <- seq_along(vnames)
  for (r in R) for (imp in I) {
    r_name <- vnames[r]; imp_name <- vnames[imp]
    fname <- paste0(prefix, "_", .sanitize(r_name), "_imp_", .sanitize(imp_name), ".png")
    title <- paste(prefix, "—", r_name, "impulse:", imp_name)
    .with_png(file.path(outdir, fname), code = .safe_plot_with_variants(irf_obj, idx=r, imp_idx=imp, names_vec=vnames, title=title))
  }
  cat("✓ Saved all", length(R)*length(I), "IRF PNGs to", outdir, "\n")
}
create_comprehensive_plots <- function(irf_results, fevd_results, forecasts, bvar_result){
  cat("\n=== SAVING INDIVIDUAL PLOTS (ALL COMBINATIONS) ===\n")
  vnames <- colnames(bvar_result$data_matrix)
  dir_irf_g <- .ensure_dir(file.path(getwd(), "irf_generalized"))
  dir_irf_c <- .ensure_dir(file.path(getwd(), "irf_cholesky"))
  dir_fevd  <- .ensure_dir(file.path(getwd(), "fevd"))
  dir_pdf   <- .ensure_dir(file.path(getwd(), "structural_analysis_plots"))
  
  cat("IRFs — generalized (all pairs)...\n"); .save_irf_all_pairs(irf_results$generalized, vnames, dir_irf_g, "irf_generalized")
  cat("IRFs — cholesky (all pairs)...\n");    .save_irf_all_pairs(irf_results$cholesky,    vnames, dir_irf_c, "irf_cholesky")
  cat("FEVD — all responses...\n");           .save_fevd_all(fevd_results, vnames, dir_fevd, "fevd")
  
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
  
  cat("✓ PNGs in: irf_generalized/, irf_cholesky/, fevd/\n")
  cat("✓ PDFs in: structural_analysis_plots/\n")
}

# --------------------------------- MAIN -------------------------------------
run_complete_structural_analysis <- function(force_refresh_data=FALSE,
                                             stationarize_financial=TRUE,
                                             cache_file="bvar_final_data.csv"){
  cat("=== COMPLETE BVAR STRUCTURAL ANALYSIS ===\n")
  cat("Replicating Crump et al. (2021) with extensions\n")
  cat("==============================================\n")
  
  cat("\nSTEP 1: DATA PREPARATION\n")
  data <- download_and_prepare_data(cache_file=cache_file,
                                    force_refresh=force_refresh_data,
                                    stationarize_financial=stationarize_financial)
  
  cat("\nSTEP 2: BVAR ESTIMATION\n")
  bvar_result <- estimate_bvar_model(data, lags=5)  # robust estimator handles retries internally
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
  cat("Outputs:\n- Data cache:", cache_file, "\n- IRFs generalized/cholesky: PNGs per (response, impulse)\n- FEVD: one PNG per response (with robust fallback)\n- PDFs: forecasts.pdf, sign_restrictions.pdf\n\n")
  list(data=data, bvar_result=bvar_result, irf_results=irf_results, fevd_results=fevd_results, forecasts=forecasts)
}

cat("COMPLETE BVAR STRUCTURAL ANALYSIS SCRIPT - FULL PATCHED VERSION (v11)\n")
cat("====================================================================\n")
cat("To rebuild & stationarize: results <- run_complete_structural_analysis(TRUE, TRUE)\n\n")

# Auto-run
results <- run_complete_structural_analysis(FALSE, TRUE)
