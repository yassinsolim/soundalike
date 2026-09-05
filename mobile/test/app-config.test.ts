import { describe, expect, test } from "vitest";

import appConfig from "../app.json";

/**
 * The iOS share extension is the whole point of the app on iOS, and it can go
 * missing without failing the build. expo-share-intent derives the Xcode target
 * name from iosShareExtensionName by stripping non-alphanumerics, then skips
 * creating the target if a target by that name already exists. Naming the
 * extension after the app therefore matched the main app target and silently
 * produced a build with no share extension in it.
 */
function sanitize(name: string): string {
  return name.replace(/[^a-zA-Z0-9]/g, "");
}

type SharePluginOptions = {
  iosShareExtensionName: string;
  iosActivationRules: Record<string, unknown>;
  androidIntentFilters: string[];
};

type AppExtension = {
  targetName: string;
  bundleIdentifier: string;
  entitlements: Record<string, string[]>;
};

const expo = appConfig.expo as unknown as {
  name: string;
  slug: string;
  ios: { bundleIdentifier: string };
  plugins: unknown[];
  updates?: { url?: string };
  runtimeVersion?: { policy?: string };
  extra: {
    eas: {
      projectId: string;
      build: { experimental: { ios: { appExtensions: AppExtension[] } } };
    };
  };
};

const sharePlugin = expo.plugins.find(
  (plugin): plugin is [string, SharePluginOptions] =>
    Array.isArray(plugin) && plugin[0] === "expo-share-intent"
);

describe("share extension config", () => {
  test("the plugin is configured", () => {
    expect(sharePlugin).toBeDefined();
  });

  test("the extension target name does not collide with the app target", () => {
    const extensionName = sharePlugin?.[1].iosShareExtensionName as string;
    expect(extensionName).toBeTruthy();
    expect(sanitize(extensionName)).not.toBe(sanitize(expo.name));
  });

  test("the EAS extension entry matches the target the plugin will create", () => {
    const extensionName = sharePlugin?.[1].iosShareExtensionName as string;
    const declared = expo.extra.eas.build.experimental.ios.appExtensions;
    expect(declared).toHaveLength(1);
    expect(declared[0].targetName).toBe(sanitize(extensionName));
  });

  test("the extension bundle id and app group follow the plugin's defaults", () => {
    const appId = expo.ios.bundleIdentifier;
    const declared = expo.extra.eas.build.experimental.ios.appExtensions[0];
    expect(declared.bundleIdentifier).toBe(`${appId}.share-extension`);
    expect(declared.entitlements["com.apple.security.application-groups"]).toEqual([
      `group.${appId}`,
    ]);
  });

  test("android still declares a text share target", () => {
    expect(sharePlugin?.[1].androidIntentFilters).toContain("text/*");
  });

  test("iOS accepts links, web pages, and plain text", () => {
    const rules = sharePlugin?.[1].iosActivationRules as Record<string, unknown>;
    expect(rules.NSExtensionActivationSupportsWebURLWithMaxCount).toBe(1);
    expect(rules.NSExtensionActivationSupportsWebPageWithMaxCount).toBe(1);
    expect(rules.NSExtensionActivationSupportsText).toBe(true);
  });
});

/**
 * Over-the-air updates replace the JavaScript in an installed app without a
 * store release, so a bad configuration here is not a build failure, it is a
 * crash on launch for people who already installed the app.
 *
 * The runtime version policy is the guard against that. Under the "appVersion"
 * policy a runtime version is just the version string, so adding a native
 * module without remembering to bump the version would let an update reach a
 * binary whose native code cannot run it. The "fingerprint" policy derives the
 * runtime version from the native project itself, including every autolinked
 * module and every config plugin, so an incompatible update cannot be
 * delivered at all.
 */
describe("over the air update config", () => {
  test("updates point at this project and nowhere else", () => {
    expect(expo.updates?.url).toBe(`https://u.expo.dev/${expo.extra.eas.projectId}`);
  });

  test("the runtime version is derived from the native project", () => {
    expect(expo.runtimeVersion?.policy).toBe("fingerprint");
  });
});
