// Private object storage on the server's local disk.
//
// Objects live under SOUNDALIKE_STORAGE_DIR using their pathname as a relative
// path, for example human-ratings/v17/<session>/<digest>.json. The interface
// (head/put/get/list/del) is what the ratings, feedback and inbox code expects.
import { randomBytes } from "node:crypto";
import { constants } from "node:fs";
import {
  link,
  lstat,
  mkdir,
  open,
  readdir,
  rename,
  unlink,
} from "node:fs/promises";
import { isAbsolute, join, relative, resolve, sep } from "node:path";

const SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const MAX_PATHNAME_LENGTH = 512;
const MAX_DEPTH = 8;
const MAX_LIST_LIMIT = 1000;
const CONTENT_TYPES = { ".json": "application/json" };
const NO_FOLLOW = constants.O_NOFOLLOW ?? 0;

export class ObjectNotFoundError extends Error {
  constructor() {
    super("The requested object does not exist.");
    this.name = "ObjectNotFoundError";
  }
}

export class ObjectExistsError extends Error {
  constructor() {
    super("An object already exists at this pathname.");
    this.name = "ObjectExistsError";
    this.code = "EEXIST";
  }
}

export function isObjectNotFound(error) {
  return (
    error instanceof ObjectNotFoundError ||
    error?.name === "ObjectNotFoundError"
  );
}

function segmentsOf(pathname) {
  if (
    typeof pathname !== "string" ||
    pathname.length === 0 ||
    pathname.length > MAX_PATHNAME_LENGTH
  ) {
    throw new TypeError("Invalid storage pathname.");
  }
  const segments = pathname.split("/");
  if (
    segments.length > MAX_DEPTH ||
    !segments.every((segment) => SEGMENT.test(segment))
  ) {
    throw new TypeError("Invalid storage pathname.");
  }
  return segments;
}

function contentTypeFor(pathname) {
  const dot = pathname.lastIndexOf(".");
  const extension = dot === -1 ? "" : pathname.slice(dot).toLowerCase();
  return CONTENT_TYPES[extension] ?? "application/octet-stream";
}

function missing(error) {
  return error?.code === "ENOENT" || error?.code === "ENOTDIR";
}

async function syncDirectory(directory) {
  let handle;
  try {
    handle = await open(directory, "r");
    await handle.sync();
  } catch {
    // Directory fsync is unsupported on some platforms (Windows); the file
    // itself is already synced.
  } finally {
    await handle?.close().catch(() => {});
  }
}

export function createDiskStorage(
  rootDirectory = () => process.env.SOUNDALIKE_STORAGE_DIR,
) {
  function root() {
    const value =
      typeof rootDirectory === "function" ? rootDirectory() : rootDirectory;
    if (typeof value !== "string" || !isAbsolute(value)) {
      throw new Error("Private storage is not configured.");
    }
    return resolve(value);
  }

  function locate(pathname) {
    const base = root();
    const segments = segmentsOf(pathname);
    const file = resolve(base, ...segments);
    const inside = relative(base, file);
    if (!inside || inside.startsWith("..") || isAbsolute(inside)) {
      throw new TypeError("Invalid storage pathname.");
    }
    return { base, file, parents: segments.slice(0, -1) };
  }

  // Walks each parent below the root with lstat so a symlinked directory can
  // never redirect an operation outside the root. With create, missing parents
  // are made one level at a time.
  async function realParents(base, parents, create = false) {
    let directory = base;
    for (const segment of parents) {
      directory = join(directory, segment);
      let metadata;
      try {
        metadata = await lstat(directory);
      } catch (error) {
        if (!missing(error)) throw error;
        if (!create) return false;
        try {
          await mkdir(directory, { mode: 0o700 });
        } catch (mkdirError) {
          if (mkdirError?.code !== "EEXIST") throw mkdirError;
        }
        metadata = await lstat(directory);
      }
      if (!metadata.isDirectory()) {
        if (create) throw new Error("Storage path is not a directory.");
        return false;
      }
    }
    return true;
  }

  async function regularFile(file) {
    let metadata;
    try {
      metadata = await lstat(file);
    } catch (error) {
      if (missing(error)) throw new ObjectNotFoundError();
      throw error;
    }
    if (!metadata.isFile()) throw new ObjectNotFoundError();
    return metadata;
  }

  async function head(pathname) {
    const { base, file, parents } = locate(pathname);
    if (!(await realParents(base, parents))) throw new ObjectNotFoundError();
    const metadata = await regularFile(file);
    return {
      pathname,
      size: metadata.size,
      contentType: contentTypeFor(pathname),
      uploadedAt: metadata.mtime,
    };
  }

  async function put(pathname, body, options = {}) {
    const { base, file, parents } = locate(pathname);
    if (options.addRandomSuffix) {
      throw new TypeError("Random pathname suffixes are not supported.");
    }
    const bytes =
      typeof body === "string" ? Buffer.from(body, "utf8") : Buffer.from(body);
    const directory = resolve(file, "..");
    await realParents(base, parents, true);
    const temporary = join(
      directory,
      `.tmp-${process.pid}-${randomBytes(8).toString("hex")}`,
    );
    const handle = await open(temporary, "wx", 0o600);
    try {
      await handle.writeFile(bytes);
      await handle.sync();
    } finally {
      await handle.close();
    }
    try {
      if (options.allowOverwrite === true) {
        await rename(temporary, file);
      } else {
        try {
          await link(temporary, file);
        } catch (error) {
          if (error?.code === "EEXIST") throw new ObjectExistsError();
          throw error;
        }
      }
      await syncDirectory(directory);
    } finally {
      await unlink(temporary).catch(() => {});
    }
    return {
      pathname,
      size: bytes.length,
      contentType: options.contentType ?? contentTypeFor(pathname),
    };
  }

  async function get(pathname) {
    const { base, file, parents } = locate(pathname);
    if (!(await realParents(base, parents))) throw new ObjectNotFoundError();
    let handle;
    try {
      handle = await open(file, constants.O_RDONLY | NO_FOLLOW);
    } catch (error) {
      if (
        missing(error) ||
        error?.code === "ELOOP" ||
        error?.code === "EISDIR"
      ) {
        throw new ObjectNotFoundError();
      }
      throw error;
    }
    let metadata;
    try {
      metadata = await handle.stat();
      if (!metadata.isFile()) throw new ObjectNotFoundError();
    } catch (error) {
      await handle.close();
      throw error;
    }
    return {
      statusCode: 200,
      stream: handle.createReadStream(),
      blob: {
        pathname,
        size: metadata.size,
        contentType: contentTypeFor(pathname),
        uploadedAt: metadata.mtime,
      },
    };
  }

  async function walk(base, directory, prefix, found) {
    let entries;
    try {
      entries = await readdir(directory, { withFileTypes: true });
    } catch (error) {
      if (missing(error)) return;
      throw error;
    }
    for (const entry of entries) {
      if (!SEGMENT.test(entry.name)) continue;
      const absolute = join(directory, entry.name);
      const pathname = relative(base, absolute).split(sep).join("/");
      if (entry.isDirectory()) {
        if (`${pathname}/`.startsWith(prefix) || prefix.startsWith(`${pathname}/`)) {
          await walk(base, absolute, prefix, found);
        }
      } else if (entry.isFile() && pathname.startsWith(prefix)) {
        found.push({ pathname, absolute });
      }
    }
  }

  async function list(options = {}) {
    const base = root();
    const prefix = options.prefix ?? "";
    if (typeof prefix !== "string" || prefix.length > MAX_PATHNAME_LENGTH) {
      throw new TypeError("Invalid storage prefix.");
    }
    const directoryPart = prefix.includes("/")
      ? prefix.slice(0, prefix.lastIndexOf("/"))
      : "";
    if (directoryPart) segmentsOf(directoryPart);
    const limit = Math.min(
      Math.max(Number.isInteger(options.limit) ? options.limit : MAX_LIST_LIMIT, 1),
      MAX_LIST_LIMIT,
    );
    const cursor = options.cursor;
    if (cursor !== undefined && typeof cursor !== "string") {
      throw new TypeError("Invalid storage cursor.");
    }
    const directorySegments = directoryPart ? directoryPart.split("/") : [];
    if (!(await realParents(base, directorySegments))) {
      return { blobs: [], hasMore: false, cursor: undefined };
    }
    const found = [];
    await walk(
      base,
      resolve(base, ...directorySegments),
      prefix,
      found,
    );
    found.sort((a, b) => (a.pathname < b.pathname ? -1 : a.pathname > b.pathname ? 1 : 0));
    const remaining = cursor === undefined
      ? found
      : found.filter((item) => item.pathname > cursor);
    const page = remaining.slice(0, limit);
    const blobs = [];
    for (const item of page) {
      const metadata = await lstat(item.absolute);
      blobs.push({
        pathname: item.pathname,
        size: metadata.size,
        uploadedAt: metadata.mtime,
      });
    }
    const hasMore = remaining.length > page.length;
    return {
      blobs,
      hasMore,
      cursor: hasMore ? page[page.length - 1].pathname : undefined,
    };
  }

  async function del(pathname) {
    const { base, file, parents } = locate(pathname);
    if (!(await realParents(base, parents))) return;
    try {
      await unlink(file);
    } catch (error) {
      if (!missing(error)) throw error;
    }
  }

  return { head, put, get, list, del };
}

const defaultStorage = createDiskStorage();

export const head = defaultStorage.head;
export const put = defaultStorage.put;
export const get = defaultStorage.get;
export const list = defaultStorage.list;
export const del = defaultStorage.del;
