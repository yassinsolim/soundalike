// Local-only server for the home page, the phone page and their APIs:
//   node tools/mobile-dev-server.js [port]        (default 8799)
// Serves / with search.js, /find, and the /api/mobile/* rewrites of site.json
// through the same dispatcher as production. /api/anysong/* is passed on to
// the engine and /api/preview asks Deezer, as the site's proxy and function
// do. Point it at an engine with SOUNDALIKE_ENGINE_URL, for example
// http://127.0.0.1:8792/api/anysong (the default is production).
import { readFileSync } from "node:fs";
import { createServer } from "node:http";

import dispatch from "../api/ratings.js";
import { DEFAULT_ENGINE_URL } from "../server/mobile.js";

const port = Number(process.argv[2] || 8799);
const engine = String(process.env.SOUNDALIKE_ENGINE_URL || DEFAULT_ENGINE_URL).replace(/\/+$/, "");
const site = JSON.parse(readFileSync(new URL("../site.json", import.meta.url), "utf8"));
const rewrites = new Map(
  site.rewrites
    .filter((rewrite) => rewrite.source.startsWith("/api/mobile/"))
    .map((rewrite) => [rewrite.source, rewrite.destination]),
);
const pages = new Map([
  ["/", ["../index.html", "text/html; charset=utf-8"]],
  ["/search.js", ["../search.js", "text/javascript; charset=utf-8"]],
  ["/clip-tap.js", ["../clip-tap.js", "text/javascript; charset=utf-8"]],
  ["/find", ["../find/index.html", "text/html; charset=utf-8"]],
  ["/find/", ["../find/index.html", "text/html; charset=utf-8"]],
]);

function sendJson(response, status, body) {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json");
  response.setHeader("Cache-Control", "no-store");
  response.end(typeof body === "string" ? body : JSON.stringify(body));
}

createServer(async (request, response) => {
  const url = new URL(request.url, "http://localhost");
  const destination = rewrites.get(url.pathname);
  if (destination) {
    const query = url.searchParams.toString();
    request.url = query ? `${destination}&${query}` : destination;
    return dispatch(request, response);
  }
  const page = pages.get(url.pathname);
  if (page) {
    response.setHeader("Content-Type", page[1]);
    response.setHeader("Cache-Control", "no-cache");
    return response.end(readFileSync(new URL(page[0], import.meta.url)));
  }
  try {
    if (url.pathname.startsWith("/api/anysong/")) {
      // A clip of the listener's audio (POST /api/anysong/excerpt) goes on with its body.
      const chunks = [];
      if (request.method === "POST") for await (const chunk of request) chunks.push(chunk);
      const answer = await fetch(
        `${engine}${url.pathname.slice("/api/anysong".length)}${url.search}`,
        request.method === "POST"
          ? {
              method: "POST",
              headers: { "Content-Type": request.headers["content-type"] || "application/octet-stream" },
              body: Buffer.concat(chunks),
            }
          : {},
      );
      return sendJson(response, answer.status, await answer.text());
    }
    if (url.pathname === "/api/preview") {
      const id = url.searchParams.get("id") || "";
      if (!/^[1-9][0-9]{0,15}$/.test(id)) return sendJson(response, 400, { ok: false, error: "bad id" });
      const track = await (await fetch(`https://api.deezer.com/track/${id}`)).json();
      return sendJson(response, 200, {
        ok: Boolean(track.preview),
        preview: track.preview || "",
        cover: track.album?.cover_medium || "",
      });
    }
  } catch {
    return sendJson(response, 502, { ok: false, error: "upstream" });
  }
  response.statusCode = 404;
  return response.end("not found");
}).listen(port, "127.0.0.1", () => {
  console.log(`dev server: http://127.0.0.1:${port}/ and /find (engine ${engine})`);
});
