/**
 * netqueue.js -- the inference scheduler, on its own so it can be tested directly.
 *
 * Inference takes 0.2-0.7 s while the controls move freely, so three things must hold:
 *
 *   1. A result is committed only if the inputs that produced it are still current.
 *      Otherwise an earlier pupil's Strehl paints itself under a later one -- the bug
 *      this module exists to prevent.
 *   2. A request that arrives while inference is running is never lost. Dropping it left
 *      the page showing a stale result with no pending work to replace it.
 *   3. No redundant run. If the in-flight attempt notices the inputs changed and redoes
 *      the work itself, that request has been absorbed -- running again afterwards would
 *      predict the same inputs twice.
 *
 * Point 3 is why completion is tracked by version rather than a boolean: a queued flag
 * cannot tell "a request arrived and was absorbed by the retry" from "a request arrived
 * after the retry started", and re-runs in both cases.
 *
 * @param {object} o
 * @param {() => any}            o.gather   snapshot the current inputs
 * @param {(inputs:any) => Promise<any>} o.predict  the slow part
 * @param {(result:any, inputs:any) => void} o.commit  apply a result known to be current
 * @param {(err:Error) => void} [o.onError]
 * @param {() => void}          [o.onPending]  called when inputs change, before the wait
 * @param {number}              [o.debounceMs=180]
 * @param {(fn:Function,ms:number)=>any} [o.setTimer]  injectable for tests
 */
export function createScheduler(o) {
  const {
    gather, predict, commit,
    onError = () => {}, onPending = () => {},
    debounceMs = 180,
    setTimer = setTimeout, clearTimer = clearTimeout,
  } = o;

  let busy = false;      // an attempt is in flight
  let timer = null;      // the debounce
  let version = 0;       // bumped by every input change
  let settled = -1;      // the version whose result was committed (or failed)

  /** Call whenever the inputs change. */
  function schedule() {
    version++;
    onPending();
    clearTimer(timer);
    /* The timer must ask whether there is still work, not just whether we are free. If the
       in-flight attempt's retry already absorbed and settled this change, a timer left over
       from it would start a second prediction for inputs that were just committed. */
    timer = setTimer(() => { if (!busy && version !== settled) start(); }, debounceMs);
  }

  async function start() {
    busy = true;
    try {
      for (;;) {
        const v = version;
        const inputs = gather();
        let result;
        try {
          result = await predict(inputs);
        } catch (err) {
          onError(err);
          settled = v;          // do not retry a failing predictor
          return;
        }
        if (v !== version) continue;   // superseded mid-flight: absorb it and redo
        try { commit(result, inputs); } catch (err) { onError(err); }
        settled = v;
        return;
      }
    } finally {
      busy = false;
      // Only if something genuinely newer arrived than what we just settled.
      if (version !== settled) start();
    }
  }

  return {
    schedule,
    /**
     * Run now, skipping the debounce -- and deliberately skipping the version check too.
     * loadNet() calls this once a session exists, at which point `settled` may already
     * equal `version` from the 'model not loaded' rejection, so a guarded call would do
     * nothing. flush() means "run regardless".
     */
    flush() { clearTimer(timer); if (!busy) start(); },
    get busy() { return busy; },
    get version() { return version; },
    get settled() { return settled; },
  };
}
