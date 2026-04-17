import {
    BehavioralLogger,
    PreAuthBehaviorCollector,
    fetchSessionSnapshot,
    startSession,
} from "./logger.js";

const SESSION_KEY = "ato-demo-session";
const RESEARCH_LABEL_KEY = "ato-demo-research-label";
const POLL_INTERVAL_MS = 2200;

const USER_BASELINES = {
    alice: {
        trustedIp: "203.0.113.18",
        trustedDevice: "alice-work-laptop",
        typicalLoginHour: 8.5,
        preferredPages: ["/home", "/profile", "/settings", "/home"],
    },
    bravo: {
        trustedIp: "203.0.113.27",
        trustedDevice: "bravo-desktop",
        typicalLoginHour: 10.0,
        preferredPages: ["/home", "/settings", "/profile", "/home"],
    },
    charlie: {
        trustedIp: "203.0.113.45",
        trustedDevice: "charlie-macbook",
        typicalLoginHour: 14.0,
        preferredPages: ["/home", "/profile", "/home", "/settings"],
    },
};

function browserName() {
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes("edg")) {
        return "edge";
    }
    if (ua.includes("firefox")) {
        return "firefox";
    }
    if (ua.includes("safari") && !ua.includes("chrome")) {
        return "safari";
    }
    if (ua.includes("chrome")) {
        return "chrome";
    }
    return "unknown-browser";
}

function osName() {
    const ua = navigator.userAgent.toLowerCase();
    if (ua.includes("windows")) {
        return "windows";
    }
    if (ua.includes("mac os")) {
        return "macos";
    }
    if (ua.includes("linux")) {
        return "linux";
    }
    return "unknown-os";
}

function defaultBaseline(userId) {
    const normalized = String(userId || "demo-user").trim().toLowerCase();
    const now = new Date();
    const currentHour = now.getHours() + (now.getMinutes() / 60);
    return (
        USER_BASELINES[normalized]
        || {
            trustedIp: "203.0.113.18",
            trustedDevice: `${normalized || "demo"}-research-browser`,
            typicalLoginHour: currentHour,
            preferredPages: ["/home", "/profile", "/settings", "/home"],
        }
    );
}

function readJson(key, fallback) {
    const raw = sessionStorage.getItem(key);
    if (!raw) {
        return fallback;
    }
    try {
        return JSON.parse(raw);
    } catch (error) {
        console.warn(`Failed to parse session storage key ${key}`, error);
        return fallback;
    }
}

function writeJson(key, value) {
    sessionStorage.setItem(key, JSON.stringify(value));
}

function readSession() {
    return readJson(SESSION_KEY, null);
}

function writeSession(nextValue) {
    writeJson(SESSION_KEY, nextValue);
}

function mergeSession(patch) {
    const current = readSession() || {};
    writeSession({ ...current, ...patch });
}

function clearSession() {
    sessionStorage.removeItem(SESSION_KEY);
    sessionStorage.removeItem(RESEARCH_LABEL_KEY);
}

async function loginUser(username, password) {
    const response = await fetch("/api/v1/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
    });

    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.error || "Login failed");
    }
    return payload;
}

function readResearchLabel() {
    return readJson(RESEARCH_LABEL_KEY, {
        mode: "LIVE",
        label: "Live user observation",
        note: "Analyst annotation only. The model receives behavior and context, not this label.",
    });
}

function setResearchLabel(mode, label, note) {
    writeJson(RESEARCH_LABEL_KEY, { mode, label, note });
    renderResearchLabel();
}

function setSlot(slot, value) {
    document.querySelectorAll(`[data-slot="${slot}"]`).forEach((element) => {
        element.textContent = value;
    });
}

function setHtml(slot, value) {
    document.querySelectorAll(`[data-slot="${slot}"]`).forEach((element) => {
        element.innerHTML = value;
    });
}

function ensureList(slot, items, emptyText) {
    document.querySelectorAll(`[data-slot="${slot}"]`).forEach((element) => {
        element.innerHTML = "";
        const values = items.length ? items : [emptyText];
        values.forEach((value) => {
            const item = document.createElement("li");
            item.textContent = value;
            element.appendChild(item);
        });
    });
}

function riskClass(riskLevel) {
    return `pill pill-risk risk-${String(riskLevel || "LOW").toLowerCase()}`;
}

function stateClass(detectedState) {
    const normalized = String(detectedState || "OBSERVING").toLowerCase().replace(/[^a-z]+/g, "-");
    return `pill pill-state state-${normalized}`;
}

function renderRiskPills(assessment) {
    document.querySelectorAll('[data-slot="risk-level"]').forEach((element) => {
        element.className = riskClass(assessment.risk_level);
        element.textContent = assessment.risk_level || "LOW";
    });
    document.querySelectorAll('[data-slot="detected-state"]').forEach((element) => {
        element.className = stateClass(assessment.detected_state);
        element.textContent = assessment.detected_state || "OBSERVING";
    });
}

function renderResearchLabel() {
    const research = readResearchLabel();
    setSlot("research-label", research.label);
    setSlot("research-note", research.note);
    document.querySelectorAll('[data-slot="research-mode"]').forEach((element) => {
        element.className = `pill pill-research mode-${String(research.mode || "LIVE").toLowerCase()}`;
        element.textContent = research.mode || "LIVE";
    });
}

function formatFeatureName(name) {
    return String(name || "")
        .replaceAll("_", " ")
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function renderAssessment(assessment) {
    if (!assessment || !assessment.session_id) {
        return;
    }

    setSlot("session-id", assessment.session_id.slice(0, 8));
    setSlot("anomaly-score", Number(assessment.anomaly_score || 0).toFixed(4));
    setSlot("raw-anomaly-score", Number(assessment.raw_anomaly_score || 0).toFixed(4));
    setSlot("combined-score", Number(assessment.combined_score || 0).toFixed(4));
    setSlot("context-deviation", Number(assessment.context_deviation || 0).toFixed(4));
    setSlot("risk-action", assessment.action || "ALLOW_SESSION");
    setSlot("window-events", String(assessment.window_event_count || 0));
    setSlot("processed-events", String(assessment.processed_events || 0));
    setSlot("score-ready", assessment.score_ready ? "READY" : "WARMING_UP");
    setSlot("score-readiness-detail", assessment.score_ready ? "Behavioral score active" : "Waiting for minimum event threshold");

    // Experiment framework: show active param count
    if (assessment.active_param_count !== undefined) {
        setSlot("active-params", `${assessment.active_param_count}/${assessment.total_param_count || 22}`);
    }

    renderRiskPills(assessment);

    const reasons = Array.isArray(assessment.reasons) ? assessment.reasons : [];
    const topFeatures = Array.isArray(assessment.top_deviation_features) ? assessment.top_deviation_features : [];
    ensureList("explanations", reasons, "No explanation available yet.");
    ensureList(
        "top-features",
        topFeatures.map((feature) => formatFeatureName(feature)),
        "Feature deviations will appear after scoring becomes active.",
    );

    document.querySelectorAll("[data-mfa-banner]").forEach((element) => {
        element.classList.toggle("active", assessment.risk_level === "HIGH");
    });

    mergeSession({ lastAssessment: assessment });
}

function formatEvent(event) {
    const parts = [event.type || "event", event.page || ""].filter(Boolean);
    if (event.type === "keystroke" && event.field_name) {
        parts.push(`field=${event.field_name}`);
    }
    if (event.type === "click" && event.hesitation !== undefined && event.hesitation !== null) {
        parts.push(`hesitation=${Number(event.hesitation).toFixed(2)}s`);
    }
    if (event.type === "mouse_move" && event.velocity !== undefined && event.velocity !== null) {
        parts.push(`velocity=${Number(event.velocity).toFixed(0)}`);
    }
    if (event.type === "page_exit" && event.dwell_time !== undefined && event.dwell_time !== null) {
        parts.push(`dwell=${Number(event.dwell_time).toFixed(2)}s`);
    }
    return parts.join(" | ");
}

function renderEventTimeline(events) {
    const container = document.getElementById("event-timeline");
    if (!container) {
        return;
    }
    container.innerHTML = "";
    const rows = Array.isArray(events) && events.length ? [...events].reverse() : [];
    if (!rows.length) {
        const item = document.createElement("li");
        item.textContent = "No event timeline yet.";
        container.appendChild(item);
        return;
    }
    rows.forEach((event) => {
        const item = document.createElement("li");
        const time = new Date(Number(event.timestamp || 0) * 1000).toLocaleTimeString();
        item.innerHTML = `<strong>${time}</strong><span>${formatEvent(event)}</span>`;
        container.appendChild(item);
    });
}

function linePoints(history, key, width, height, padding) {
    const usableWidth = width - (padding * 2);
    const usableHeight = height - (padding * 2);
    return history.map((entry, index) => {
        const x = padding + ((usableWidth * index) / Math.max(history.length - 1, 1));
        const raw = Number(entry[key] || 0);
        const y = height - padding - (Math.max(Math.min(raw, 1), 0) * usableHeight);
        return `${x},${y}`;
    }).join(" ");
}

function renderScoreGraph(history) {
    const container = document.getElementById("score-graph");
    if (!container) {
        return;
    }

    const rows = Array.isArray(history) ? history : [];
    if (!rows.length) {
        container.innerHTML = '<div class="graph-empty">No assessments yet.</div>';
        return;
    }

    const width = 760;
    const height = 240;
    const padding = 26;
    const grid = [0.25, 0.5, 0.75].map((level) => {
        const y = height - padding - (level * (height - (padding * 2)));
        return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" class="graph-grid" />`;
    }).join("");

    container.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" class="graph-svg" role="img" aria-label="Real-time anomaly score graph">
            ${grid}
            <polyline points="${linePoints(rows, "combined_score", width, height, padding)}" class="graph-line graph-combined" />
            <polyline points="${linePoints(rows, "anomaly_score", width, height, padding)}" class="graph-line graph-anomaly" />
            <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" class="graph-axis" />
            <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" class="graph-axis" />
        </svg>
    `;
}

function renderContext(snapshot) {
    if (!snapshot) {
        return;
    }
    setSlot("event-count", String(snapshot.event_count || 0));
    setSlot("device-fingerprint", snapshot.device?.fingerprint || snapshot.device?.device_id || "-");
    setSlot("context-ip", snapshot.ip_address || "-");
    setSlot("context-browser", snapshot.device?.browser || "-");
    setSlot("context-os", snapshot.device?.os || "-");
}

function renderSnapshot(snapshot) {
    if (!snapshot || !snapshot.assessment) {
        return;
    }
    renderAssessment(snapshot.assessment);
    renderScoreGraph(snapshot.assessment_history || []);
    renderEventTimeline(snapshot.recent_events || []);
    renderContext(snapshot);
}

function applyNavState(pageKey) {
    document.querySelectorAll("[data-page-link]").forEach((element) => {
        element.classList.toggle("active", element.dataset.pageLink === pageKey);
    });
}

async function safeHydrateSnapshot(session) {
    if (!session?.sessionId) {
        return null;
    }
    try {
        const snapshot = await fetchSessionSnapshot(session.sessionId);
        renderSnapshot(snapshot);
        return snapshot;
    } catch (error) {
        console.error("Failed to hydrate session snapshot", error);
        return null;
    }
}

function bindLogout(logger, poller) {
    document.querySelectorAll("[data-logout]").forEach((button) => {
        button.addEventListener("click", async (event) => {
            event.preventDefault();
            if (poller) {
                window.clearInterval(poller);
            }
            await logger.endSession();
            clearSession();
            window.location.href = "/";
        });
    });
}

function bindScenarioButtons(logger, onAfterSimulation = () => {}) {
    document.querySelectorAll("[data-simulate]").forEach((button) => {
        button.addEventListener("click", async () => {
            const mode = button.dataset.simulate || "NORMAL";
            const label = mode === "NORMAL"
                ? "Normal baseline replay"
                : `${mode} attacker replay`;
            const note = mode === "NORMAL"
                ? "Analyst replay only. The backend still infers risk from event patterns and context."
                : "Synthetic attacker profile for research validation. The backend is not told this label.";

            setResearchLabel(mode, label, note);
            button.disabled = true;
            try {
                const assessment = await logger.simulateScenario(mode);
                if (assessment?.session_id) {
                    renderAssessment(assessment);
                }
                onAfterSimulation();
            } finally {
                button.disabled = false;
            }
        });
    });
}

function buildLoginPayload(form) {
    const formData = new FormData(form);
    const userId = String(formData.get("user_id") || "alice").trim().toLowerCase();
    const baseline = defaultBaseline(userId);
    const now = new Date();

    return {
        user_id: userId,
        login_time: now.toISOString(),
        ip_address: baseline.trustedIp,
        device: {
            device_id: baseline.trustedDevice,
            device_type: "desktop",
            os: osName(),
            browser: browserName(),
            user_agent: navigator.userAgent,
        },
        profile: {
            typical_login_hour: baseline.typicalLoginHour,
            trusted_ips: [baseline.trustedIp],
            trusted_devices: [baseline.trustedDevice.toLowerCase()],
            preferred_pages: baseline.preferredPages,
        },
    };
}

export function initializeLoginPage() {
    const form = document.getElementById("login-form");
    const baselineModal = document.getElementById("baseline-modal");
    const baselineSubmitBtn = document.getElementById("baseline-submit-btn");
    const baselineTextInput = document.getElementById("baseline-typing-input");
    const baselineScrollContainer = document.getElementById("baseline-scroll-container");

    if (!form) {
        return;
    }

    const preview = document.getElementById("session-preview");
    const collector = new PreAuthBehaviorCollector({ currentPage: () => "/login" });
    collector.start();
    renderResearchLabel();

    async function completeLogin(payload) {
        const submitButton = form.querySelector('button[type="submit"]');
        try {
            const password = String(new FormData(form).get("password") || "");
            await loginUser(payload.user_id, password);
            payload.bootstrap_events = collector.finish({ nextPage: "/home" });
            const response = await startSession(payload);
            writeSession({
                sessionId: response.session_id,
                userId: payload.user_id,
                lastAssessment: response.assessment,
            });
            setResearchLabel(
                "LIVE",
                "Live user observation",
                "Analyst annotation only. The model still receives only behavior and context.",
            );

            if (preview) {
                preview.textContent = JSON.stringify(response, null, 2);
            }

            window.location.href = "/home";
        } catch (error) {
            console.error("Failed to start session", error);
            if (preview) {
                preview.textContent = `Failed to start session: ${error}`;
            }
            if (submitButton) {
                submitButton.disabled = false;
            }
        }
    }

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const payload = buildLoginPayload(form);
        const submitButton = form.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
        }

        // Show enrollment modal
        if (baselineModal) {
            baselineModal.classList.add("active");
            
            // Basic validation for modal controls
            let scrolled = false;
            let targetTyped = false;
            const targetText = "The quick brown fox jumps over the lazy dog.";

            // Reset
            baselineTextInput.value = "";
            baselineSubmitBtn.disabled = true;

            baselineScrollContainer.addEventListener("scroll", () => {
                if (baselineScrollContainer.scrollTop + baselineScrollContainer.clientHeight >= baselineScrollContainer.scrollHeight - 10) {
                    scrolled = true;
                    checkEnrollment();
                }
            });

            baselineTextInput.addEventListener("input", () => {
                if (baselineTextInput.value === targetText) {
                    targetTyped = true;
                    checkEnrollment();
                } else {
                    targetTyped = false;
                    checkEnrollment();
                }
            });

            function checkEnrollment() {
                if (scrolled && targetTyped) {
                    baselineSubmitBtn.disabled = false;
                } else {
                    baselineSubmitBtn.disabled = true;
                }
            }

            baselineSubmitBtn.onclick = async () => {
                baselineSubmitBtn.disabled = true;
                baselineSubmitBtn.textContent = "Storing Baseline...";
                
                // Drain any events from the collector during this modal
                const baselineEvents = collector.drain();

                try {
                    await fetch("/api/v1/sessions/enroll", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            user_id: payload.user_id,
                            events: baselineEvents
                        }),
                    });
                } catch (e) {
                    console.error("Failed to post baseline", e);
                }

                baselineModal.classList.remove("active");
                await completeLogin(payload);
            };
        } else {
            // Fallback if modal DOM missing
            completeLogin(payload);
        }
    });
}

function ensureSessionOrRedirect() {
    const session = readSession();
    if (!session?.sessionId) {
        window.location.href = "/";
        return null;
    }
    return session;
}

function renderStoredAssessment() {
    const session = readSession();
    if (session?.lastAssessment) {
        renderAssessment(session.lastAssessment);
    }
    renderResearchLabel();
}

function buildLogger(session, pageKey) {
    return new BehavioralLogger({
        sessionId: session.sessionId,
        username: session.userId,
        currentPage: () => pageKey,
        onAssessment: (assessment) => {
            if (assessment?.session_id) {
                renderAssessment(assessment);
            }
        },
    });
}

function bindTextHelpers(pageKey) {
    if (pageKey === "/home") {
        const seedButton = document.getElementById("seed-search");
        const searchInput = document.getElementById("workspace-search");
        seedButton?.addEventListener("click", () => {
            if (searchInput) {
                searchInput.value = "travel reimbursement workflow";
                searchInput.focus();
            }
        });
    }

    if (pageKey === "/profile") {
        const button = document.getElementById("load-profile-draft");
        button?.addEventListener("click", () => {
            const fields = {
                display_name: "Alice Security Analyst",
                title: "Fraud Operations Analyst",
                location: "Jakarta",
                bio: "Behavioral biometrics research prototype reviewer.",
            };
            Object.entries(fields).forEach(([name, value]) => {
                const input = document.querySelector(`[name="${name}"]`);
                if (input instanceof HTMLInputElement || input instanceof HTMLTextAreaElement) {
                    input.value = value;
                }
            });
        });
    }

    if (pageKey === "/settings") {
        const button = document.getElementById("load-device-template");
        button?.addEventListener("click", () => {
            const fields = {
                device_alias: "trusted-windows-station",
                recovery_email: "alice@example.com",
            };
            Object.entries(fields).forEach(([name, value]) => {
                const input = document.querySelector(`[name="${name}"]`);
                if (input instanceof HTMLInputElement) {
                    input.value = value;
                }
            });
        });
    }
}

function bindDashboardLinks() {
    document.querySelectorAll("[data-open-dashboard]").forEach((element) => {
        element.addEventListener("click", (event) => {
            event.preventDefault();
            window.location.href = "/dashboard";
        });
    });
}

function bindTransactionAction(logger) {
    const payButton = document.getElementById("pay-button");
    const payStatus = document.getElementById("pay-status");
    if (!payButton) {
        return;
    }

    payButton.addEventListener("click", async () => {
        payButton.disabled = true;
        if (payStatus) {
            payStatus.textContent = "Processing payment...";
        }

        try {
            await logger.completeTransaction();
            clearSession();
            if (payStatus) {
                payStatus.textContent = "Payment complete. Session ended.";
            }
            window.setTimeout(() => {
                window.location.href = "/";
            }, 800);
        } catch (error) {
            console.error("Failed to complete transaction", error);
            if (payStatus) {
                payStatus.textContent = "Payment failed. Please try again.";
            }
            payButton.disabled = false;
        }
    });
}

export function initializeMonitoredPage(pageKey) {
    const session = ensureSessionOrRedirect();
    if (!session) {
        return;
    }

    applyNavState(pageKey);
    renderStoredAssessment();

    const logger = buildLogger(session, pageKey);
    logger.start();
    bindDashboardLinks();
    bindTextHelpers(pageKey);
    bindTransactionAction(logger);
    bindScenarioButtons(logger, () => {
        void safeHydrateSnapshot(session);
    });

    const poller = window.setInterval(() => {
        void safeHydrateSnapshot(session);
    }, POLL_INTERVAL_MS);

    bindLogout(logger, poller);
    void safeHydrateSnapshot(session);
}

export function initializeDashboard() {
    const session = ensureSessionOrRedirect();
    if (!session) {
        return;
    }

    applyNavState("/dashboard");
    renderStoredAssessment();

    const logger = buildLogger(session, "/dashboard");
    logger.start();
    bindDashboardLinks();

    const refresh = () => {
        void safeHydrateSnapshot(session);
    };

    bindScenarioButtons(logger, () => {
        refresh();
        void refreshExperimentLog();
    });
    const poller = window.setInterval(refresh, POLL_INTERVAL_MS);
    bindLogout(logger, poller);
    refresh();

    // ── Experiment Framework UI ──
    void initExperimentToggles();
    void refreshExperimentLog();
    document.getElementById("btn-refresh-log")?.addEventListener("click", () => {
        void refreshExperimentLog();
    });
}

// ===========================================================================
// EXPERIMENT TOGGLE UI
// ===========================================================================

/**
 * Parameter metadata for grouping and labelling in the toggle UI.
 * Each entry: [paramKey, displayLabel, reserved?]
 * reserved = true means the feature is not yet collected by the frontend
 * and the toggle has no runtime effect (displayed greyed-out for transparency).
 */
const TOGGLE_META = [
    // group label, params[]
    [
        "⌨️ Keystroke Dynamics",
        [
            ["dwell_time",         "Dwell Time",         false],
            ["flight_time",        "Flight Time",        false],
            ["typing_consistency", "Typing Consistency", false],
            ["error_rate",         "Error Rate",         true],   // reserved
        ],
    ],
    [
        "🖱️ Mouse & Pointer",
        [
            ["mouse_trajectory",   "Mouse Trajectory",  false],
            ["mouse_velocity",     "Mouse Velocity",    false],
            ["mouse_acceleration", "Mouse Acceleration",false],
            ["click_interval",     "Click Interval",    false],
            ["scroll_behavior",    "Scroll Behavior",   false],
        ],
    ],
    [
        "👆 Touch / Gesture",
        [
            ["swipe_speed",    "Swipe Speed",    true],  // mobile reserved
            ["gesture_pattern","Gesture Pattern",true],
        ],
    ],
    [
        "🌐 Context & Environment",
        [
            ["device_fingerprint","Device Fingerprint",false],
            ["ip_geo",            "IP / Geolocation",   false],
            ["login_time",        "Login Time",         false],
            ["session_duration",  "Session Duration",   false],
        ],
    ],
    [
        "🗺️ Navigation Behavior",
        [
            ["navigation_pattern","Navigation Pattern",false],
            ["time_per_page",     "Time per Page",     false],
            ["action_sequence",   "Action Sequence",   false],
        ],
    ],
    [
        "🧠 Cognitive Signals",
        [
            ["decision_latency",  "Decision Latency", false],
            ["hesitation",        "Hesitation",        false],
        ],
    ],
    [
        "🖥️ Browser-Level",
        [
            ["tab_switching",  "Tab Switching",   true],  // reserved
            ["idle_time",      "Idle Time",        true],
            ["clipboard_usage","Clipboard Usage",  true],
        ],
    ],
];

/** Presets — sets of parameter keys to enable */
const PRESETS = {
    all: null,   // null = enable all
    minimal: ["mouse_velocity", "mouse_trajectory", "click_interval",
              "dwell_time", "flight_time", "typing_consistency",
              "device_fingerprint", "ip_geo", "login_time"],
    keystroke: ["dwell_time", "flight_time", "typing_consistency"],
    mouse:     ["mouse_trajectory", "mouse_velocity", "mouse_acceleration",
                "click_interval", "scroll_behavior"],
};

/** Build the toggle UI from TOGGLE_META and populate with server values */
async function initExperimentToggles() {
    const container = document.getElementById("toggle-groups");
    if (!container) {
        return;
    }

    // Fetch current configuration from server
    let currentToggles = {};
    try {
        const resp = await fetch("/api/v1/experiment/toggles");
        const data = await resp.json();
        currentToggles = data.toggles || {};
        updateActiveBadge(data.active_count, data.total_count);
    } catch (err) {
        console.warn("Could not fetch experiment toggles", err);
    }

    container.innerHTML = "";

    // Build toggle groups
    TOGGLE_META.forEach(([groupLabel, params]) => {
        const groupEl = document.createElement("div");
        groupEl.className = "toggle-group";

        const labelEl = document.createElement("div");
        labelEl.className = "toggle-group-label";
        labelEl.textContent = groupLabel;
        groupEl.appendChild(labelEl);

        const grid = document.createElement("div");
        grid.className = "toggle-grid";

        params.forEach(([key, displayLabel, reserved]) => {
            const item = document.createElement("div");
            item.className = "toggle-item";

            const switchEl = document.createElement("label");
            switchEl.className = "toggle-switch";
            switchEl.title = reserved ? `${displayLabel} (reserved — not yet collected)` : displayLabel;

            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.id = `toggle-${key}`;
            checkbox.dataset.toggleKey = key;
            checkbox.checked = Boolean(currentToggles[key]);
            if (reserved) {
                checkbox.disabled = true;
            }

            // Update active badge on any change
            checkbox.addEventListener("change", updateActiveBadgeFromCheckboxes);

            const track = document.createElement("span");
            track.className = "toggle-track";
            switchEl.appendChild(checkbox);
            switchEl.appendChild(track);

            const lbl = document.createElement("label");
            lbl.htmlFor = `toggle-${key}`;
            lbl.className = "toggle-label" + (reserved ? " reserved" : "");
            lbl.textContent = reserved ? `${displayLabel} ✦` : displayLabel;

            item.appendChild(switchEl);
            item.appendChild(lbl);
            grid.appendChild(item);
        });

        groupEl.appendChild(grid);
        container.appendChild(groupEl);
    });

    // Apply button
    document.getElementById("btn-apply-toggles")?.addEventListener("click", applyToggles);

    // Preset buttons
    bindPresetButton("btn-preset-all",       PRESETS.all);
    bindPresetButton("btn-preset-minimal",   PRESETS.minimal);
    bindPresetButton("btn-preset-keystroke", PRESETS.keystroke);
    bindPresetButton("btn-preset-mouse",     PRESETS.mouse);

    updateActiveBadgeFromCheckboxes();
}

function bindPresetButton(id, enabledSet) {
    document.getElementById(id)?.addEventListener("click", () => {
        document.querySelectorAll("[data-toggle-key]").forEach((cb) => {
            if (cb.disabled) {
                return;
            }
            if (enabledSet === null) {
                cb.checked = true;   // enable all
            } else {
                cb.checked = enabledSet.includes(cb.dataset.toggleKey);
            }
        });
        updateActiveBadgeFromCheckboxes();
    });
}

function updateActiveBadgeFromCheckboxes() {
    let active = 0;
    let total = 0;
    document.querySelectorAll("[data-toggle-key]").forEach((cb) => {
        total += 1;
        if (cb.checked) {
            active += 1;
        }
    });
    updateActiveBadge(active, total);
}

function updateActiveBadge(active, total) {
    const label = document.getElementById("active-count-label");
    const bar   = document.getElementById("coverage-bar");
    if (label) {
        label.textContent = `${active} / ${total} features active`;
    }
    if (bar) {
        const pct = total > 0 ? Math.round((active / total) * 100) : 100;
        bar.style.width = `${pct}%`;
    }
}

async function applyToggles() {
    const btn = document.getElementById("btn-apply-toggles");
    const statusEl = document.getElementById("toggle-status");
    if (btn) {
        btn.disabled = true;
    }
    if (statusEl) {
        statusEl.textContent = "Applying…";
        statusEl.style.color = "#93c5fd";
    }

    const updates = {};
    document.querySelectorAll("[data-toggle-key]").forEach((cb) => {
        updates[cb.dataset.toggleKey] = cb.checked;
    });

    try {
        const resp = await fetch("/api/v1/experiment/toggles", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(updates),
        });
        const data = await resp.json();
        const hasErrors = data.errors && Object.keys(data.errors).length > 0;
        if (statusEl) {
            if (hasErrors) {
                statusEl.textContent = `Applied with errors: ${JSON.stringify(data.errors)}`;
                statusEl.style.color = "#fca5a5";
            } else {
                statusEl.textContent = `✓ Applied — ${data.active_count}/${data.total_count} features active`;
                statusEl.style.color = "#86efac";
            }
        }
        updateActiveBadge(data.active_count, data.total_count);
    } catch (err) {
        if (statusEl) {
            statusEl.textContent = `Error: ${err.message}`;
            statusEl.style.color = "#fca5a5";
        }
    } finally {
        if (btn) {
            btn.disabled = false;
        }
        // Clear status after 4 s
        setTimeout(() => {
            if (statusEl) {
                statusEl.textContent = "";
            }
        }, 4000);
    }
}

// ===========================================================================
// EXPERIMENT LOG TABLE
// ===========================================================================

async function refreshExperimentLog() {
    try {
        const resp = await fetch("/api/v1/experiment/log?n=10");
        const data = await resp.json();
        renderExperimentLog(data.records || []);
        const countEl = document.getElementById("exp-log-count");
        if (countEl) {
            countEl.textContent = `${data.count} recent records`;
        }
    } catch (err) {
        console.warn("Could not load experiment log", err);
    }
}

function renderExperimentLog(records) {
    const tbody = document.getElementById("exp-log-body");
    if (!tbody) {
        return;
    }

    if (!records.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="exp-empty">No experiment records yet. Run a scenario to generate data.</td></tr>';
        return;
    }

    // Show newest first
    const rows = [...records].reverse();
    tbody.innerHTML = "";

    rows.forEach((rec) => {
        const tr = document.createElement("tr");
        const ts = rec.timestamp ? new Date(rec.timestamp).toLocaleTimeString() : "—";
        const sid = rec.session_id ? rec.session_id.slice(0, 8) : "—";
        const activeStr = `${rec.active_param_count ?? "?"}/${rec.total_param_count ?? 22}`;
        const anomaly = Number(rec.anomaly_score || 0).toFixed(3);
        const raw_anomaly = Number(rec.raw_anomaly_score || 0).toFixed(3);
        const combined = Number(rec.combined_score || 0).toFixed(3);
        const risk = rec.risk_level || "—";
        const decision = rec.decision || "—";

        const riskCls  = `risk-${ risk.toLowerCase()}`;
        const decCls   = `decision-${decision.toLowerCase()}`;

        tr.innerHTML = [
            `<td>${ts}</td>`,
            `<td title="${rec.session_id || ""}"><code>${sid}</code></td>`,
            `<td>${activeStr}</td>`,
            `<td title="Raw Score: ${raw_anomaly}">${anomaly} <span style="font-size:0.8em;color:var(--text-soft)">(${raw_anomaly})</span></td>`,
            `<td>${combined}</td>`,
            `<td class="${riskCls}">${risk}</td>`,
            `<td class="${decCls}">${decision}</td>`,
        ].join("");
        tbody.appendChild(tr);
    });
}
