const DEFAULT_SCENARIOS = {
    NORMAL: {
        label: "Normal baseline replay",
        pages: ["/home", "/profile", "/settings", "/home"],
        speed: 760,
        speedJitter: 140,
        hesitation: [0.28, 0.46, 0.38, 0.42],
        typing: { interval: 0.18, intervalJitter: 0.05, hold: 0.11, holdJitter: 0.03 },
        transitionDelay: [1.2, 1.8, 1.4, 1.6],
        clickIntervalBase: 1.05,
        linearity: 0.35,
    },
    BOT: {
        label: "BOT attacker replay",
        pages: ["/settings", "/profile", "/settings"],
        speed: 1950,
        speedJitter: 0,
        hesitation: [0, 0, 0],
        typing: { interval: 0.03, intervalJitter: 0.002, hold: 0.015, holdJitter: 0.001 },
        transitionDelay: [0.08, 0.06, 0.08],
        clickIntervalBase: 0.08,
        linearity: 0.98,
    },
    HUMAN: {
        label: "Human attacker replay",
        pages: ["/home", "/settings", "/home", "/profile"],
        speed: 1220,
        speedJitter: 360,
        hesitation: [0.05, 0.09, 0.04, 0.08],
        typing: { interval: 0.09, intervalJitter: 0.05, hold: 0.055, holdJitter: 0.025 },
        transitionDelay: [0.32, 0.24, 0.18, 0.28],
        clickIntervalBase: 0.22,
        linearity: 0.62,
    },
    FAST: {
        label: "Fast attacker replay",
        pages: ["/home", "/profile", "/settings", "/profile"],
        speed: 2320,
        speedJitter: 220,
        hesitation: [0.015, 0.02, 0.015, 0.02],
        typing: { interval: 0.045, intervalJitter: 0.01, hold: 0.022, holdJitter: 0.006 },
        transitionDelay: [0.06, 0.05, 0.05, 0.05],
        clickIntervalBase: 0.09,
        linearity: 0.82,
    },
};

function nowSeconds() {
    return Date.now() / 1000;
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function sampleGaussian(meanValue, spread) {
    if (!spread) {
        return meanValue;
    }
    const u = 1 - Math.random();
    const v = 1 - Math.random();
    const standard = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return meanValue + (standard * spread);
}

function readFieldName(target) {
    if (!(target instanceof HTMLElement)) {
        return "";
    }
    return (
        target.getAttribute("name")
        || target.id
        || target.dataset.trackField
        || target.getAttribute("aria-label")
        || target.getAttribute("placeholder")
        || target.tagName.toLowerCase()
    );
}

function keyIdentity(event) {
    return `${event.code}:${event.location}:${event.key}`;
}

function formatScenarioText(mode, page) {
    const normalized = String(page || "/home").replace("/", "").toLowerCase() || "home";
    if (mode === "BOT") {
        return normalized === "settings" ? "mfa-reset" : "export";
    }
    if (mode === "HUMAN") {
        return normalized === "profile" ? "verify owner" : "change-now";
    }
    if (mode === "FAST") {
        return normalized === "profile" ? "rush" : "otp";
    }
    if (normalized === "home") {
        return "quarterly security memo";
    }
    if (normalized === "profile") {
        return "team analyst";
    }
    return "trusted-workstation";
}

class BaseCollector {
    constructor({
        currentPage = () => window.location.pathname,
        mouseThrottleMs = 40,
    } = {}) {
        this.currentPage = currentPage;
        this.mouseThrottleMs = mouseThrottleMs;
        this.buffer = [];
        this.lastMouseSampleAt = 0;
        this.lastMouseEvent = null;
        this.previousAngle = null;
        this.lastClickTime = null;
        this.lastKeyTime = null;
        this.lastBehaviorAt = null;
        this.lastScrollTop = window.scrollY || 0;
        this.keydownMap = new Map();
        this.pageEnteredAt = null;
        this.pageEnteredPath = "";
        this.pageExitSent = false;
        this.boundHandlers = {};
    }

    start() {
        this.boundHandlers.mousemove = (event) => this.handleMouseMove(event);
        this.boundHandlers.click = (event) => this.handleClick(event);
        this.boundHandlers.keydown = (event) => this.handleKeydown(event);
        this.boundHandlers.keyup = (event) => this.handleKeyup(event);
        this.boundHandlers.scroll = () => this.handleScroll();
        this.boundHandlers.focusin = (event) => this.handleFocusIn(event);

        document.addEventListener("mousemove", this.boundHandlers.mousemove, { passive: true });
        document.addEventListener("click", this.boundHandlers.click, { passive: true });
        document.addEventListener("keydown", this.boundHandlers.keydown);
        document.addEventListener("keyup", this.boundHandlers.keyup);
        document.addEventListener("focusin", this.boundHandlers.focusin);
        window.addEventListener("scroll", this.boundHandlers.scroll, { passive: true });

        this.trackPageEnter(this.currentPage());
    }

    detach() {
        document.removeEventListener("mousemove", this.boundHandlers.mousemove);
        document.removeEventListener("click", this.boundHandlers.click);
        document.removeEventListener("keydown", this.boundHandlers.keydown);
        document.removeEventListener("keyup", this.boundHandlers.keyup);
        document.removeEventListener("focusin", this.boundHandlers.focusin);
        window.removeEventListener("scroll", this.boundHandlers.scroll);
    }

    trackPageEnter(page = this.currentPage(), timestamp = nowSeconds()) {
        this.pageEnteredAt = timestamp;
        this.pageEnteredPath = page;
        this.pageExitSent = false;
        this.enqueue({
            type: "page_enter",
            timestamp,
            page,
        });
    }

    trackPageExit(reason = "navigation", timestamp = nowSeconds()) {
        if (this.pageEnteredAt === null || this.pageExitSent) {
            return;
        }
        this.pageExitSent = true;
        this.enqueue({
            type: "page_exit",
            timestamp,
            page: this.pageEnteredPath || this.currentPage(),
            dwell_time: Math.max(timestamp - this.pageEnteredAt, 0),
            reason,
        });
    }

    noteBehavior(timestamp) {
        this.lastBehaviorAt = timestamp;
    }

    handleFocusIn(event) {
        const timestamp = nowSeconds();
        this.noteBehavior(timestamp);
        const fieldName = readFieldName(event.target);
        if (fieldName) {
            this.enqueue({
                type: "focus",
                timestamp,
                page: this.currentPage(),
                field_name: fieldName,
            });
        }
    }

    handleMouseMove(event) {
        const nowMs = Date.now();
        if (nowMs - this.lastMouseSampleAt < this.mouseThrottleMs) {
            return;
        }
        this.lastMouseSampleAt = nowMs;
        const timestamp = nowMs / 1000;
        let velocity = 0;
        let acceleration = 0;
        let trajectoryAngle = 0;
        let directionChange = 0;

        if (this.lastMouseEvent) {
            const deltaSeconds = Math.max(timestamp - this.lastMouseEvent.timestamp, 0.001);
            const deltaX = event.clientX - this.lastMouseEvent.x;
            const deltaY = event.clientY - this.lastMouseEvent.y;
            const distance = Math.hypot(deltaX, deltaY);
            velocity = distance / deltaSeconds;
            acceleration = (velocity - this.lastMouseEvent.velocity) / deltaSeconds;
            trajectoryAngle = Math.atan2(deltaY, deltaX);
            directionChange = this.previousAngle === null ? 0 : Math.abs(trajectoryAngle - this.previousAngle);
            this.previousAngle = trajectoryAngle;
        }

        this.lastMouseEvent = {
            x: event.clientX,
            y: event.clientY,
            timestamp,
            velocity,
        };
        this.noteBehavior(timestamp);
        this.enqueue({
            type: "mouse_move",
            timestamp,
            page: this.currentPage(),
            x: event.clientX,
            y: event.clientY,
            velocity,
            acceleration,
            trajectory_angle: trajectoryAngle,
            direction_change: directionChange,
        });
    }

    handleClick(event) {
        const timestamp = nowSeconds();
        const clickInterval = this.lastClickTime === null ? null : timestamp - this.lastClickTime;
        const hesitationReference = this.lastBehaviorAt ?? this.pageEnteredAt ?? timestamp;
        const hesitation = Math.max(timestamp - hesitationReference, 0);
        this.lastClickTime = timestamp;
        this.noteBehavior(timestamp);
        this.enqueue({
            type: "click",
            timestamp,
            page: this.currentPage(),
            x: event.clientX,
            y: event.clientY,
            click_interval: clickInterval,
            hesitation,
            field_name: readFieldName(event.target),
        });
    }

    handleKeydown(event) {
        if (event.repeat) {
            return;
        }
        const timestamp = nowSeconds();
        this.keydownMap.set(keyIdentity(event), {
            timestamp,
            fieldName: readFieldName(event.target),
        });
        this.noteBehavior(timestamp);
    }

    handleKeyup(event) {
        const timestamp = nowSeconds();
        const identity = keyIdentity(event);
        const keydown = this.keydownMap.get(identity);
        const keyInterval = this.lastKeyTime === null ? null : timestamp - this.lastKeyTime;
        const holdTime = keydown ? Math.max(timestamp - keydown.timestamp, 0) : 0;

        this.enqueue({
            type: "keystroke",
            timestamp,
            page: this.currentPage(),
            key_interval: keyInterval,
            hold_time: holdTime,
            field_name: keydown?.fieldName || readFieldName(event.target),
            key: event.key,
        });

        this.lastKeyTime = timestamp;
        this.noteBehavior(timestamp);
        this.keydownMap.delete(identity);
    }

    handleScroll() {
        const timestamp = nowSeconds();
        const currentTop = window.scrollY || 0;
        const delta = currentTop - this.lastScrollTop;
        this.lastScrollTop = currentTop;
        this.noteBehavior(timestamp);
        this.enqueue({
            type: "scroll",
            timestamp,
            page: this.currentPage(),
            scroll_delta: delta,
        });
    }

    enqueue(event) {
        this.buffer.push(event);
    }

    drain() {
        return this.buffer.splice(0, this.buffer.length);
    }
}

export class PreAuthBehaviorCollector extends BaseCollector {
    finish({ nextPage = "/home" } = {}) {
        const timestamp = nowSeconds();
        this.trackPageExit("login_submit", timestamp);
        this.enqueue({
            type: "navigation",
            timestamp,
            page: nextPage,
        });
        this.detach();
        return this.drain();
    }
}

export class BehavioralLogger extends BaseCollector {
    constructor({
        sessionId,
        username,
        apiBaseUrl = window.location.origin,
        flushIntervalMs = 1400,
        mouseThrottleMs = 40,
        currentPage = () => window.location.pathname,
        onAssessment = () => {},
    }) {
        super({ currentPage, mouseThrottleMs });
        this.sessionId = sessionId;
        this.username = username;
        this.apiBaseUrl = apiBaseUrl;
        this.flushIntervalMs = flushIntervalMs;
        this.onAssessment = onAssessment;
        this.flushTimer = null;
        this.inFlight = false;
        this.ending = false;
    }

    start() {
        super.start();
        this.boundHandlers.visibilitychange = () => {
            if (document.visibilityState === "hidden") {
                this.enqueue({
                    type: "tab_switch",
                    timestamp: nowSeconds(),
                    page: this.currentPage()
                });
                this.flush({ useBeacon: true });
            }
        };
        this.boundHandlers.pagehide = () => {
            this.trackPageExit("pagehide");
            this.flush({ useBeacon: true });
        };
        
        // IDLE TRACKING
        this.lastActivity = Date.now();
        this.boundHandlers.activity = () => {
            this.lastActivity = Date.now();
        };
        document.addEventListener("mousemove", this.boundHandlers.activity, { passive: true });
        document.addEventListener("keydown", this.boundHandlers.activity, { passive: true });
        document.addEventListener("click", this.boundHandlers.activity, { passive: true });

        this.idleTimer = window.setInterval(() => {
            let idle = (Date.now() - this.lastActivity) / 1000;
            if (idle > 2) {
                this.enqueue({
                    type: "idle",
                    idle_time: idle,
                    timestamp: nowSeconds(),
                    page: this.currentPage()
                });
                // Prevent continuous logging every second while idle
                this.lastActivity = Date.now();
            }
        }, 1000);

        // CLIPBOARD
        this.boundHandlers.paste = () => {
            this.enqueue({
                type: "clipboard",
                action: "paste",
                timestamp: nowSeconds(),
                page: this.currentPage()
            });
        };
        this.boundHandlers.copy = () => {
            this.enqueue({
                type: "clipboard",
                action: "copy",
                timestamp: nowSeconds(),
                page: this.currentPage()
            });
        };
        document.addEventListener("paste", this.boundHandlers.paste);
        document.addEventListener("copy", this.boundHandlers.copy);

        document.addEventListener("visibilitychange", this.boundHandlers.visibilitychange);
        window.addEventListener("pagehide", this.boundHandlers.pagehide);

        this.flushTimer = window.setInterval(() => {
            void this.flush();
        }, this.flushIntervalMs);
    }

    detach() {
        if (this.flushTimer) {
            window.clearInterval(this.flushTimer);
            this.flushTimer = null;
        }
        if (this.idleTimer) {
            window.clearInterval(this.idleTimer);
            this.idleTimer = null;
        }
        document.removeEventListener("visibilitychange", this.boundHandlers.visibilitychange);
        window.removeEventListener("pagehide", this.boundHandlers.pagehide);
        document.removeEventListener("mousemove", this.boundHandlers.activity);
        document.removeEventListener("keydown", this.boundHandlers.activity);
        document.removeEventListener("click", this.boundHandlers.activity);
        document.removeEventListener("paste", this.boundHandlers.paste);
        document.removeEventListener("copy", this.boundHandlers.copy);
        super.detach();
    }

    async endSession() {
        if (this.ending) {
            return null;
        }
        this.ending = true;
        this.trackPageExit("session_end");
        this.enqueue({
            type: "session_end",
            timestamp: nowSeconds(),
            page: this.currentPage(),
        });
        const assessment = await this.flush();
        this.detach();
        const response = await fetch(`${this.apiBaseUrl}/api/v1/sessions/${this.sessionId}/end`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: this.username,
            }),
        });
        if (!response.ok) {
            throw new Error("Failed to end session");
        }
        return assessment;
    }

    async completeTransaction() {
        if (this.ending) {
            return null;
        }
        this.ending = true;
        this.trackPageExit("transaction_pay");
        this.enqueue({
            type: "session_end",
            timestamp: nowSeconds(),
            page: this.currentPage(),
        });
        await this.flush();
        this.detach();

        const response = await fetch(`${this.apiBaseUrl}/api/v1/transaction/pay`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: this.username,
                session_id: this.sessionId,
            }),
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "Failed to complete transaction");
        }
        if (payload?.assessment) {
            this.onAssessment(payload.assessment);
        }
        return payload;
    }

    flush({ useBeacon = false } = {}) {
        if (!this.sessionId || this.buffer.length === 0) {
            return Promise.resolve(null);
        }

        if (useBeacon) {
            const events = this.drain();
            const sent = navigator.sendBeacon(
                `${this.apiBaseUrl}/api/v1/events`,
                new Blob(
                    [
                        JSON.stringify({
                            username: this.username,
                            session_id: this.sessionId,
                            events,
                        }),
                    ],
                    { type: "application/json" },
                ),
            );
            if (!sent) {
                this.buffer.unshift(...events);
            }
            return Promise.resolve(null);
        }

        if (this.inFlight) {
            return Promise.resolve(null);
        }

        this.inFlight = true;
        const events = this.drain();

        return fetch(`${this.apiBaseUrl}/api/v1/events`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: this.username,
                session_id: this.sessionId,
                events,
            }),
        })
            .then(async (response) => {
                const assessment = await response.json();
                this.onAssessment(assessment);
                return assessment;
            })
            .catch((error) => {
                console.error("Failed to flush behavioral events", error);
                this.buffer.unshift(...events);
                return null;
            })
            .finally(() => {
                this.inFlight = false;
            });
    }

    buildScenarioEvents(mode = "NORMAL") {
        const scenario = DEFAULT_SCENARIOS[mode] || DEFAULT_SCENARIOS.NORMAL;
        const events = [];
        let timestamp = nowSeconds();
        let x = this.lastMouseEvent?.x ?? 180;
        let y = this.lastMouseEvent?.y ?? 180;
        let clickTimestamp = this.lastClickTime ?? (timestamp - scenario.clickIntervalBase);
        let keyTimestamp = this.lastKeyTime ?? (timestamp - scenario.typing.interval);

        scenario.pages.forEach((page, pageIndex) => {
            timestamp += scenario.transitionDelay[Math.min(pageIndex, scenario.transitionDelay.length - 1)];
            events.push({
                type: "navigation",
                timestamp,
                page,
            });
            events.push({
                type: "page_enter",
                timestamp,
                page,
            });

            const steps = mode === "BOT" ? 8 : mode === "FAST" ? 6 : 10;
            const targetX = 240 + (pageIndex * 180) + (mode === "BOT" ? 420 : 0);
            const targetY = 160 + (pageIndex * 90);
            let previousAngle = null;

            for (let step = 1; step <= steps; step += 1) {
                const progress = step / steps;
                const baseX = x + ((targetX - x) * progress);
                const baseY = y + ((targetY - y) * progress);
                const wobble = (1 - scenario.linearity) * 26;
                const offsetX = mode === "BOT" ? 0 : Math.sin(progress * Math.PI * 3) * wobble;
                const offsetY = mode === "FAST"
                    ? Math.cos(progress * Math.PI * 5) * (wobble * 0.5)
                    : Math.cos(progress * Math.PI * 2) * wobble;
                const nextX = clamp(baseX + offsetX, 32, 1320);
                const nextY = clamp(baseY + offsetY, 48, 760);
                const velocity = Math.max(sampleGaussian(scenario.speed, scenario.speedJitter), 40);
                const acceleration = mode === "BOT"
                    ? 0
                    : Math.max(sampleGaussian(mode === "FAST" ? 2800 : 980, mode === "FAST" ? 360 : 240), 0);
                const angle = Math.atan2(nextY - y, nextX - x);
                const directionChange = previousAngle === null ? 0 : Math.abs(angle - previousAngle);
                previousAngle = angle;
                const deltaSeconds = Math.max(
                    Math.hypot(nextX - x, nextY - y) / velocity,
                    mode === "BOT" ? 0.012 : 0.04,
                );
                timestamp += deltaSeconds;
                x = nextX;
                y = nextY;
                events.push({
                    type: "mouse_move",
                    timestamp,
                    page,
                    x,
                    y,
                    velocity,
                    acceleration,
                    trajectory_angle: angle,
                    direction_change: directionChange,
                });
            }

            const hesitation = scenario.hesitation[Math.min(pageIndex, scenario.hesitation.length - 1)];
            timestamp += hesitation;
            const clickInterval = timestamp - clickTimestamp;
            clickTimestamp = timestamp;
            events.push({
                type: "click",
                timestamp,
                page,
                x,
                y,
                click_interval: clickInterval,
                hesitation,
            });

            if (page !== "/dashboard") {
                const fieldName = page === "/home" ? "search" : page === "/profile" ? "display_name" : "device_alias";
                const text = formatScenarioText(mode, page);
                for (const character of text) {
                    const keyGap = Math.max(sampleGaussian(scenario.typing.interval, scenario.typing.intervalJitter), 0.02);
                    const holdTime = Math.max(sampleGaussian(scenario.typing.hold, scenario.typing.holdJitter), 0.01);
                    timestamp += keyGap;
                    events.push({
                        type: "keystroke",
                        timestamp,
                        page,
                        key_interval: timestamp - keyTimestamp,
                        hold_time: holdTime,
                        field_name: fieldName,
                        key: character,
                    });
                    keyTimestamp = timestamp;
                    timestamp += holdTime;
                }
            }

            const scrollBursts = mode === "BOT" ? 0 : page === "/home" ? 3 : 1;
            for (let index = 0; index < scrollBursts; index += 1) {
                timestamp += mode === "FAST" ? 0.03 : 0.18;
                events.push({
                    type: "scroll",
                    timestamp,
                    page,
                    scroll_delta: page === "/home" ? 240 : 120,
                });
            }

            const dwellTime = mode === "BOT"
                ? 0.08
                : Math.max(
                    scenario.transitionDelay[Math.min(pageIndex, scenario.transitionDelay.length - 1)]
                    + (mode === "NORMAL" ? 0.8 : 0.12),
                    0.05,
                );
            timestamp += dwellTime;
            events.push({
                type: "page_exit",
                timestamp,
                page,
                dwell_time: dwellTime,
            });
        });

        return events;
    }

    async simulateScenario(mode = "NORMAL") {
        const events = this.buildScenarioEvents(mode);
        events.forEach((event) => this.enqueue(event));

        // Wait for any in-flight flush to complete before sending scenario events.
        // Without this, the periodic timer flush can hold the inFlight lock,
        // causing flush() to silently return null and leave events buffered.
        const maxWait = 12;
        let waited = 0;
        while (this.inFlight && waited < maxWait) {
            await new Promise((resolve) => { setTimeout(resolve, 250); });
            waited += 1;
        }

        return this.flush();
    }
}

export async function startSession(payload, apiBaseUrl = window.location.origin) {
    const response = await fetch(`${apiBaseUrl}/api/v1/sessions/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    return response.json();
}

export async function fetchSessionSnapshot(sessionId, apiBaseUrl = window.location.origin) {
    const response = await fetch(`${apiBaseUrl}/api/v1/sessions/${sessionId}`);
    return response.json();
}
