// Local-only server for the phone page and its API:
//   node tools/mobile-dev-server.js [port]        (default 8799)
// Serves /find and the /api/mobile/* rewrites of site.json through the same
// dispatcher as production. Point it at an engine with SOUNDALIKE_ENGINE_URL,
// for example http://127.0.0.1:8792/api/anysong (the default is production).
import { readFileSync } from "node:fs";
import { createServer } from "node:http";

import dispatch from "../api/ratings.js";

const port = Number(process.argv[2] || 8799);
const site = JSON.parse(readFileSync(new URL("../site.json", import.meta.url), "utf8"));
const rewrites = new Map(
  site.rewrites
    .filter((rewrite) => rewrite.source.startsWith("/api/mobile/"))
    .map((rewrite) => [rewrite.source, rewrite.destination]),
);

createServer((request, response) => {
  const url = new URL(request.url, "http://localhost");
  const destination = rewrites.get(url.pathname);
  if (destination) {
    const query = url.searchParams.toString();
    request.url = query ? `${destination}&${query}` : destination;
    return dispatch(request, response);
  }
  if (url.pathname === "/find" || url.pathname === "/find/") {
    response.setHeader("Content-Type", "text/html; charset=utf-8");
    response.setHeader("Cache-Control", "no-cache");
    return response.end(readFileSync(new URL("../find/index.html", import.meta.url)));
  }
  response.statusCode = 404;
  return response.end("not found");
}).listen(port, "127.0.0.1", () => {
  console.log(`mobile dev server: http://127.0.0.1:${port}/find`);
});
