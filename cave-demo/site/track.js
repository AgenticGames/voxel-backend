/*
 * Engagement tracking for cave.playmithril.com.
 *
 * COPY. The source of truth is website-edits/playmithril.com/track.js, in the
 * private playmithril.com repo. This file is a verbatim copy of it: fix bugs
 * there first, then copy across. The two are deliberately identical so the
 * demo and the marketing site produce the same event vocabulary and can be
 * read with one query.
 *
 * cave.playmithril.com is a subdomain of playmithril.com and therefore sits in
 * the same Cloudflare zone, so /px/* beacons from here land in the same edge
 * logs. Tell them apart with clientRequestHTTPHost, or by page/demo.
 *
 * Demo specific milestones (did the WASM boot, did they generate, did they
 * mine, did they run Dormancy) are fired by app.js through window.mithrilTrack.
 *
 * Every event is a GET to /px/<category>/<detail>. Those paths have no file
 * behind them, so they return 404 by design. The 404 is irrelevant: Cloudflare
 * logs the request path either way, and that log is the entire point. Read the
 * results by grouping httpRequestsAdaptiveGroups on clientRequestPath.
 *
 * Two edge rules constrain how this can talk to the network:
 *   - The WAF issues a managed challenge to any non-GET request, so beacons
 *     must be GET. That rules out navigator.sendBeacon, which is always POST.
 *   - Rate limiting blocks at 60 requests / 10s per IP, and a normal page load
 *     already spends ~15 of those on assets. Hence the queue and the caps below.
 *
 * No cookies, no localStorage, no IDs, no email addresses. State lives in
 * memory for the life of the page and dies with it.
 */
(function () {
    'use strict';

    // Global Privacy Control is legally binding in some jurisdictions. Cheap to
    // honour, and we lose almost nothing by doing it.
    if (navigator.globalPrivacyControl === true) return;

    // Raised from 30 when the press kit went up. That page has nine sections,
    // thirty copy buttons and forty downloadable assets, so a journalist working
    // through it properly could exhaust a thirty event budget before reaching
    // the downloads, and the events lost would be the only ones worth having.
    // The rate limiter is unaffected: it caps requests per ten seconds, and
    // MIN_GAP_MS already holds beacons to 25 per ten seconds no matter how many
    // are queued. A page load spends about 15 more on assets, so the ceiling
    // stays comfortably under the 60 that triggers a block.
    var MAX_EVENTS = 60;      // per page load, excluding the exit burst
    var MIN_GAP_MS = 400;     // spacing between queued beacons
    var SECTION_DWELL_MS = 2000; // time in view before a section counts as "read"

    // Assets and the press kit live on their own hostname. Without naming it,
    // every screenshot, logo and zip on the press page reports as one
    // undifferentiated ext/downloads-playmithril-com click.
    var DOWNLOAD_HOST = 'downloads.playmithril.com';
    var FILE_RE = /\.(zip|mp4|png|jpe?g|webp|gif|pdf|psd|svg|wav|mp3)$/i;

    // Destinations worth naming. Everything else falls through to ext/<host>,
    // which is enough to notice a link nobody thought to name. The demo is in
    // here deliberately: cave.playmithril.com is the one link on the site that
    // leads to the game itself, so it cannot sit in the same bucket as a
    // footer link to Unreal.
    var GOALS = [
        ['store.steampowered.com', 'steam'],
        ['cave.playmithril.com', 'demo'],
        ['discord.gg', 'discord'],
        ['discord.com', 'discord'],
        ['youtube.com', 'youtube'],
        ['youtu.be', 'youtube'],
        ['tiktok.com', 'tiktok'],
        ['github.com', 'github'],
        ['crates.io', 'crates'],
        ['x.com', 'x'],
        ['agentic-games.com', 'studio']
    ];

    var sent = 0;
    var queue = [];
    var draining = false;

    function fire(path) {
        var url = '/px/' + path;
        try {
            fetch(url, {
                method: 'GET',
                mode: 'no-cors',
                cache: 'no-store',
                keepalive: true,
                // Same-origin credentials, deliberately. Bot Fight Mode is on,
                // and omitting them strips Cloudflare's own __cf_bm cookie,
                // which makes every beacon look like unattributed automation
                // and invites a challenge. The site sets no cookies of its own.
                credentials: 'same-origin'
            }).catch(function () {});
        } catch (e) {
            // Older browsers, or fetch unavailable during unload.
            new Image().src = url + '?n=' + Math.random().toString(36).slice(2);
        }
    }

    function drain() {
        if (draining || !queue.length) return;
        draining = true;
        fire(queue.shift());
        setTimeout(function () {
            draining = false;
            drain();
        }, MIN_GAP_MS);
    }

    // Normal events go through the queue so a burst of scrolling can't trip
    // the rate limiter.
    function track(path) {
        if (sent >= MAX_EVENTS) return;
        sent++;
        queue.push(path);
        drain();
    }

    // Exit events skip the queue: there is no "later" to defer them to.
    // Deduped by path, because a keepalive request torn down mid-flight during
    // unload can be retried at the transport layer and land in the logs twice.
    var firedNow = {};
    function trackNow(path) {
        if (firedNow[path]) return;
        firedNow[path] = true;
        fire(path);
    }

    function slug(s) {
        return String(s || '')
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-+|-+$/g, '')
            .slice(0, 40) || 'none';
    }

    // Filename without its directory or extension, percent decoded, so
    // "/kit/Video%20Clips/MP4/mithril_mining.mp4" reads back as "mithril-mining".
    function fileName(src) {
        var last = String(src || '').split(/[?#]/)[0].split('/').pop();
        try { last = decodeURIComponent(last); } catch (e) {}
        return last.replace(/\.[a-z0-9]+$/i, '');
    }

    // Where on the page a click happened. The /go/ links carry their placement
    // in the URL because the redirect rules need it there, but everything else
    // has to be inferred, and the enclosing section is both accurate and free
    // of hand-tagging. Anything added to the page later is covered automatically.
    function placementOf(el) {
        if (!el.closest) return 'body';
        var sec = el.closest('section[id]');
        if (sec) return slug(sec.id);
        if (el.closest('footer')) return 'footer';
        if (el.closest('nav') || el.closest('header')) return 'nav';
        return 'body';
    }

    function goalFor(host) {
        for (var i = 0; i < GOALS.length; i++) {
            if (host === GOALS[i][0] || host.slice(-(GOALS[i][0].length + 1)) === '.' + GOALS[i][0]) {
                return GOALS[i][1];
            }
        }
        return '';
    }

    var fired = {};
    function once(key, path) {
        if (fired[key]) return;
        fired[key] = true;
        track(path);
    }

    // The only public surface. An app that has its own milestones, rather than
    // scroll depth and section dwell, reports them through here. Used by the
    // browser demo, where the events that matter are "did the WASM boot" and
    // "did they generate a cave", neither of which this file can observe.
    //
    //     mithrilTrack('demo/ready/5-15')          fires once
    //     mithrilTrack('demo/mine', {repeat: true}) fires every time
    window.mithrilTrack = function (path, opts) {
        path = String(path || '').replace(/^\/+|^px\//, '');
        if (!path) return;
        if (opts && opts.repeat) track(path);
        else once('api-' + path, path);
    };

    document.addEventListener('DOMContentLoaded', function () {

        // --- Which page this is ----------------------------------------------
        // Every event is /px/<category>/<detail>, with nowhere to put a path, so
        // without this the support page is indistinguishable from the homepage.
        // That is the gap that hid the only /support/ hit we have on record.
        // Fired first, so it is the one event guaranteed to clear the queue,
        // the event cap and the rate limiter.
        // A page can name itself by setting window.MITHRIL_PAGE before this
        // file loads. The browser demo needs that: it is served from the root
        // of cave.playmithril.com, so the path alone would report it as "home"
        // and put it in the same bucket as the marketing homepage.
        var page = location.pathname.replace(/^\/+|\/+$/g, '');
        page = window.MITHRIL_PAGE ? slug(window.MITHRIL_PAGE) : (page ? slug(page) : 'home');
        track('page/' + page);

        // --- Where they came from -------------------------------------------
        // Cloudflare already logs the referrer, but campaign tags only exist in
        // the URL, and they are the only way to tell TikTok from Reddit from a
        // Discord post. Captured once, then attached to conversion events below.
        var params = new URLSearchParams(location.search);
        var campaign = params.get('utm_source') || params.get('ref') || params.get('src') || '';
        campaign = campaign ? slug(campaign) : '';

        if (campaign) {
            track('src/' + campaign);
            var medium = params.get('utm_campaign') || params.get('utm_medium');
            if (medium) track('campaign/' + campaign + '/' + slug(medium));
        }

        // Viewport bucket. Device type is already in Cloudflare's data, but not
        // the actual width, which is what decides whether the layout works.
        var w = window.innerWidth;
        var size = w < 480 ? 'xs' : w < 768 ? 'sm' : w < 1100 ? 'md' : w < 1600 ? 'lg' : 'xl';
        track('view/' + size);

        // Deep link straight to a section, e.g. a pinned comment linking #playtest.
        if (location.hash && location.hash.length > 1) {
            track('entry/' + slug(location.hash.slice(1)));
        }

        // --- Scroll depth ----------------------------------------------------
        var marks = [25, 50, 75, 100];
        var maxPct = 0;

        function onScroll() {
            var docH = document.documentElement.scrollHeight - window.innerHeight;
            var pct = docH > 0 ? (window.scrollY / docH) * 100 : 100;
            if (pct > maxPct) maxPct = pct;
            for (var i = 0; i < marks.length; i++) {
                if (pct >= marks[i]) once('scroll' + marks[i], 'scroll/' + marks[i]);
            }
        }
        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll();

        // --- Which sections actually got read --------------------------------
        // A section counts only after it has held the middle of the viewport
        // for a couple of seconds. Scrolling past at speed is not reading.
        var sections = document.querySelectorAll('section[id]');
        var order = [];
        var dwell = {};
        var enteredAt = {};

        sections.forEach(function (el) { order.push(el.id); });

        // Where the exit summary points before any section has been read. This
        // was hardcoded to 'hero', which only exists on the homepage, so any
        // other page reported a hero bounce it could not physically have had.
        // First section if the page has one, otherwise the page itself.
        var fallbackSection = order.length ? order[0] : page;
        var deepest = fallbackSection;
        var current = fallbackSection;

        if ('IntersectionObserver' in window) {
            var secObserver = new IntersectionObserver(function (entries) {
                entries.forEach(function (entry) {
                    var id = entry.target.id;
                    if (entry.isIntersecting) {
                        enteredAt[id] = Date.now();
                        current = id;
                        if (order.indexOf(id) > order.indexOf(deepest)) deepest = id;
                        setTimeout(function () {
                            if (enteredAt[id]) once('sec-' + id, 'section/' + slug(id));
                        }, SECTION_DWELL_MS);
                    } else if (enteredAt[id]) {
                        dwell[id] = (dwell[id] || 0) + (Date.now() - enteredAt[id]);
                        enteredAt[id] = 0;
                    }
                });
            }, {
                // Watch a single line across the middle of the viewport instead
                // of asking for half the section to be on screen. The old
                // threshold of 0.5 quietly made tall sections unmeasurable: most
                // of these run two or three screens high on a phone, so "half
                // visible" is arithmetically impossible there and they never
                // registered at all. Since most real visitors arrive on mobile,
                // that silently removed almost every section from the data. This
                // fires when a section crosses the centre line, whatever its
                // height, and marks exactly one section current at a time.
                rootMargin: '-50% 0px -50% 0px',
                threshold: 0
            });

            sections.forEach(function (el) { secObserver.observe(el); });
        }

        // --- Interactions ----------------------------------------------------

        // Outbound clicks. The /go/ redirects already log these, but they cannot
        // see which campaign the visitor arrived on. This closes that loop:
        // it answers "which channel actually produces wishlists".
        document.addEventListener('click', function (e) {
            var a = e.target.closest && e.target.closest('a[href]');
            if (!a) return;
            var href = a.getAttribute('href') || '';

            if (href.indexOf('/go/') === 0) {
                var parts = href.split('/').filter(Boolean); // go, dest, placement
                var dest = slug(parts[1]);
                var placement = slug(parts[2] || 'unknown');
                track('out/' + dest + '/' + placement);
                if (campaign) trackNow('convert/' + campaign + '/' + dest);
                return;
            }

            if (href.indexOf('#') === 0 && href.length > 1) {
                track('nav/' + slug(href.slice(1)));
                return;
            }

            if (href.indexOf('mailto:') === 0) {
                // Which inbox. press@, creators@ and business@ are three
                // completely different kinds of visitor, and lumping them into
                // one mailto/click threw that away.
                track('mailto/' + slug(href.slice(7).split('@')[0]));
                return;
            }

            if (!/^https?:\/\//.test(href)) return;

            var host = '', urlPath = '';
            try {
                var u = new URL(href, location.href);
                host = u.hostname.replace(/^www\./, '');
                urlPath = u.pathname;
            } catch (err) { host = 'other'; }

            if (host === location.hostname.replace(/^www\./, '')) return;

            var where = placementOf(a);

            // Press kit assets and anything with a file extension. Which asset a
            // journalist actually takes is the single most useful thing the
            // press page can tell us, and it is invisible in the edge logs
            // beyond a raw hit count with no idea who or from where.
            if (host === DOWNLOAD_HOST || FILE_RE.test(urlPath) || a.hasAttribute('download')) {
                var name = fileName(urlPath) || 'unnamed';
                track('download/' + slug(name));
                if (campaign) trackNow('convert/' + campaign + '/download');
                return;
            }

            // Named destinations, with the section they were clicked from. This
            // is what makes an off-page Steam or demo click comparable to the
            // /go/ ones, which only exist for links we remembered to wrap.
            var goal = goalFor(host);
            if (goal) {
                track('out/' + goal + '/' + where);
                if (campaign) trackNow('convert/' + campaign + '/' + goal);
                return;
            }

            track('ext/' + slug(host));
        }, true);

        // Trailer plays, by video. Tells us whether the hero trailer or the
        // in-page ones are the thing people actually watch.
        document.querySelectorAll('.trailer[data-video-id]').forEach(function (el) {
            el.addEventListener('click', function () {
                var id = el.dataset.videoId;
                if (!id || id === 'YOUR_YOUTUBE_VIDEO_ID') return;
                once('vid-' + id, 'video/' + slug(id));
            });
        });

        // The Dormancy before/after slider. It is the one thing on the page a
        // visitor can play with, so using it is a strong interest signal.
        document.querySelectorAll('.compare').forEach(function (compare) {
            var range = compare.querySelector('.compare-range');
            if (!range) return;
            var id = compare.id || 'compare';
            range.addEventListener('input', function () {
                once('slider-' + id, 'slider/' + slug(id));
            });
        });

        // Which screenshots get opened full size.
        document.querySelectorAll('.gallery-item').forEach(function (item) {
            item.addEventListener('click', function () {
                var img = item.querySelector('img');
                var src = item.dataset.full || (img && img.getAttribute('src')) || '';
                var name = src.split('/').pop().replace(/\.[a-z0-9]+$/i, '');
                track('shot/' + slug(name));
            });
        });

        // Native video, as opposed to the YouTube facade above. The press kit
        // previews are muted and preload="metadata", so the edge log shows a
        // range request for all ten whether or not anyone pressed play. Only
        // this event can tell a preload apart from a viewing.
        document.querySelectorAll('video').forEach(function (v) {
            v.addEventListener('play', function () {
                var name = fileName(v.currentSrc || v.getAttribute('src')) || 'video';
                once('vid-' + name, 'video/' + slug(name));
            });
        });

        // Copy buttons, by what was copied. The press page has thirty of them,
        // one per boilerplate block, description length, feature, angle and
        // quote. Which one a journalist lifts tells us which pitch actually
        // works, and the old copy/click threw exactly that away. Both class
        // names are here because the homepage and the press kit were built
        // months apart and disagree about it.
        document.querySelectorAll('.copy-btn[data-copy], .cb-btn[data-copy]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var id = btn.getAttribute('data-copy');
                once('copy-' + id, 'copy/' + slug(id));
            });
        });

        // Light mode on the press kit. Small, but it is a deliberate act on a
        // page whose entire audience is people deciding whether to cover us.
        var themeBtn = document.getElementById('kitTheme');
        if (themeBtn) {
            themeBtn.addEventListener('click', function () {
                once('theme', 'theme/toggle');
            });
        }

        // --- Signup forms ----------------------------------------------------
        // The funnel that matters. Every step is separated so an abandon can be
        // told apart from a bounce, and a captcha block from a real submit.
        var formState = {};

        // script.js shows the success message the moment you submit, then quietly
        // rewrites it if the request to MailerLite fails. A visitor who sees that
        // rewrite thinks they signed up and did not, so it is worth catching.
        // The text is the only signal: the failure path reuses the same element.
        function watchOutcome(which) {
            var stop = Date.now() + 12000;
            var timer = setInterval(function () {
                var msgs = document.querySelectorAll('.newsletter-success');
                for (var i = 0; i < msgs.length; i++) {
                    if (/went wrong/i.test(msgs[i].textContent || '')) {
                        clearInterval(timer);
                        track('form/' + which + '/failed');
                        return;
                    }
                }
                if (Date.now() > stop) clearInterval(timer);
            }, 1500);
        }

        document.querySelectorAll('.signup-form').forEach(function (form) {
            var which = slug(form.id || 'form').replace(/-form$/, '');
            formState[which] = 'none';

            var email = form.querySelector('input[type="email"]');

            if (email) {
                email.addEventListener('focus', function () {
                    formState[which] = 'focused';
                    once('f-focus-' + which, 'form/' + which + '/focus');
                }, { once: true });

                // They typed something that looks like a real address. If they
                // then leave without submitting, that is a lost signup, not a
                // curious click.
                email.addEventListener('blur', function () {
                    if (email.value.indexOf('@') > 0) {
                        if (formState[which] !== 'submitted') formState[which] = 'typed';
                        once('f-typed-' + which, 'form/' + which + '/typed');
                    }
                });
            }

            form.addEventListener('submit', function () {
                var captcha = form.querySelector('[name="g-recaptcha-response"]');
                if (captcha && !captcha.value.trim()) {
                    // Blocked before it ever reached MailerLite. If this number
                    // is high, the captcha is eating signups.
                    formState[which] = 'captcha';
                    track('form/' + which + '/captcha-block');
                    return;
                }
                formState[which] = 'submitted';
                track('form/' + which + '/submit');
                if (campaign) trackNow('convert/' + campaign + '/' + which);
                watchOutcome(which);
            });
        });

        var started = Date.now();

        function bucketFor(secs) {
            return secs < 5 ? '0-5'
                : secs < 15 ? '5-15'
                : secs < 30 ? '15-30'
                : secs < 60 ? '30-60'
                : secs < 120 ? '60-120'
                : secs < 300 ? '120-300'
                : '300plus';
        }

        // --- Heartbeat --------------------------------------------------------
        // Milestones sent while the page is still alive, through the normal
        // queue. The exit summary below is a single request at the worst possible
        // moment in a page's life, so it cannot be the only source of timing: if
        // it is lost, /px/alive/60 still proves the visitor stayed a minute.
        // 600 and 900 added for the press kit. Reading the whole thing properly
        // takes longer than five minutes, so the old top milestone put a
        // skim-reader and a journalist writing from it in the same bucket. The
        // end/ buckets are deliberately left alone so they stay comparable with
        // everything recorded before today.
        var MILESTONES = [15, 30, 60, 120, 300, 600, 900];
        var beat = null;

        function startHeartbeat() {
            if (beat) clearInterval(beat);
            beat = setInterval(function () {
                var secs = Math.round((Date.now() - started) / 1000);
                for (var i = 0; i < MILESTONES.length; i++) {
                    if (secs >= MILESTONES[i]) once('alive' + MILESTONES[i], 'alive/' + MILESTONES[i]);
                }
                if (secs >= MILESTONES[MILESTONES.length - 1]) {
                    clearInterval(beat);
                    beat = null;
                }
            }, 5000);
        }

        startHeartbeat();

        // --- Exit ------------------------------------------------------------
        // One request, not four.
        //
        // The previous version fired dwell, deepest, exit and stickiest as four
        // back-to-back beacons on pagehide. Only the first two ever arrived: the
        // browser tears the page down before the rest are dispatched, so exit
        // point and stickiest section were never recorded once in the entire life
        // of this file. The whole summary now travels in one path, and a single
        // surviving request carries the complete session:
        //
        //     /px/end/<dwell>/<deepest>/<exit>/<stickiest>
        //
        // Split it on "/" to read it. "none" means the field had no value.
        //
        // pagehide and visibilitychange both matter: mobile Safari frequently
        // never fires the former.
        function onExit() {
            var secs = Math.round((Date.now() - started) / 1000);

            // Close out whichever section is still on screen so its time counts,
            // then leave it running: the visitor may come back to this same page.
            var now = Date.now();
            Object.keys(enteredAt).forEach(function (id) {
                if (enteredAt[id]) {
                    dwell[id] = (dwell[id] || 0) + (now - enteredAt[id]);
                    enteredAt[id] = now;
                }
            });

            var best = '', bestMs = 0;
            Object.keys(dwell).forEach(function (id) {
                if (dwell[id] > bestMs) { bestMs = dwell[id]; best = id; }
            });

            // trackNow dedupes by path, so tabbing away and back only reports
            // again if the visitor actually got deeper or stayed materially
            // longer. An identical summary is silently dropped.
            trackNow('end/' + bucketFor(secs)
                + '/' + slug(deepest)
                + '/' + slug(current)
                + '/' + (bestMs > 3000 ? slug(best) : 'none'));

            // A visitor who typed an email and left without submitting is the
            // most recoverable loss on the whole site.
            Object.keys(formState).forEach(function (which) {
                if (formState[which] === 'typed' || formState[which] === 'focused') {
                    trackNow('form/' + which + '/abandon-' + formState[which]);
                }
            });
        }

        window.addEventListener('pagehide', onExit);
        document.addEventListener('visibilitychange', function () {
            if (document.visibilityState === 'hidden') onExit();
        });

        // --- Back/forward cache -------------------------------------------------
        // A page restored from bfcache never re-runs DOMContentLoaded, so the old
        // version recorded nothing at all for it while Cloudflare's own beacon
        // recorded some of them. That is why the two counts disagreed. Treat a
        // restore as what it is, a fresh page view, and reset the session with it.
        window.addEventListener('pageshow', function (e) {
            if (!e.persisted) return;

            sent = 0;
            queue.length = 0;
            Object.keys(fired).forEach(function (k) { delete fired[k]; });
            Object.keys(firedNow).forEach(function (k) { delete firedNow[k]; });

            started = Date.now();
            dwell = {};
            enteredAt = {};

            // Was hardcoded to 'hero', the same bug that was fixed for the
            // first load and missed here. On the press page there is no hero
            // section, so every back-button return reported a bounce off a
            // section that does not exist on the page.
            deepest = fallbackSection;
            current = fallbackSection;

            track('page/' + page);
            track('view/' + size);
            startHeartbeat();
        });
    });
})();
