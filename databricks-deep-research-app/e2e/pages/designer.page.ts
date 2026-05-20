import { type Locator, type Page, expect } from '@playwright/test';

/**
 * Page Object Model for the Agent Designer (/designer and /designer/:id).
 *
 * Locator strategy:
 *  - Prefer aria-label / role (buttons in AgentDesignerListPage use aria-label="Edit {name}",
 *    header inputs use aria-label="Agent name" / "Agent description", Save uses aria-label="Save agent").
 *  - Fall back to data-testid where present (BlockEditor "Add Root" button has
 *    data-testid="add-root-button").
 *  - The AddToolDialog uses a Radix Dialog (role="dialog") with a labelled name input
 *    (id="tool-name-input", label text "Tool name") and a submit button with text "Add Tool".
 *  - Kind picker buttons inside AddToolDialog are plain <button> elements whose visible text is
 *    the tool kind label (e.g., "Web Search"). We match them with getByRole('button', { name }).
 */
export class DesignerPage {
  constructor(public readonly page: Page) {}

  // ---------------------------------------------------------------------------
  // Navigation
  // ---------------------------------------------------------------------------

  async navigateToList(): Promise<void> {
    await this.page.goto('/designer');
  }

  async navigateToNew(): Promise<void> {
    await this.page.goto('/designer/new');
  }

  // ---------------------------------------------------------------------------
  // List page locators
  // ---------------------------------------------------------------------------

  /** "Create New" button — appears in header and in the empty-state placeholder. */
  get createNewButton(): Locator {
    return this.page.getByRole('button', { name: /^create new$/i }).first();
  }

  /** Empty-state message when no agents exist. */
  get emptyStateText(): Locator {
    return this.page.getByText(/no agents yet/i);
  }

  /**
   * Edit button for a specific agent row (uses aria-label="Edit {name}").
   * @param agentName The name of the agent to edit.
   */
  editButton(agentName: string): Locator {
    return this.page.getByRole('button', { name: new RegExp(`^edit ${agentName}$`, 'i') });
  }

  /**
   * Delete button for a specific agent row (uses aria-label="Delete {name}").
   * @param agentName The name of the agent to delete.
   */
  deleteButton(agentName: string): Locator {
    return this.page.getByRole('button', { name: new RegExp(`^delete ${agentName}$`, 'i') });
  }

  // ---------------------------------------------------------------------------
  // Editor header locators (AgentDesignerPage)
  // ---------------------------------------------------------------------------

  /** Agent name input — aria-label="Agent name". */
  get nameInput(): Locator {
    return this.page.getByLabel('Agent name');
  }

  /** Agent description input — aria-label="Agent description". */
  get descriptionInput(): Locator {
    return this.page.getByLabel('Agent description');
  }

  /** Save button — aria-label="Save agent". */
  get saveButton(): Locator {
    return this.page.getByRole('button', { name: /^save$/i });
  }

  /** "Saved" status indicator (green text shown when !isDirty). */
  get savedIndicator(): Locator {
    return this.page.getByText(/^saved$/i);
  }

  /** "Unsaved changes" status indicator (amber text shown when isDirty). */
  get unsavedIndicator(): Locator {
    return this.page.getByText(/unsaved changes/i);
  }

  // ---------------------------------------------------------------------------
  // BlockEditor locators
  // ---------------------------------------------------------------------------

  /**
   * "Add Root" button — data-testid="add-root-button".
   * Visible only when ast === null (empty workflow state).
   */
  get addRootButton(): Locator {
    return this.page.getByTestId('add-root-button');
  }

  // ---------------------------------------------------------------------------
  // ToolsPanel locators
  // ---------------------------------------------------------------------------

  /** "Add Tool" button in the ToolsPanel header. */
  get addToolButton(): Locator {
    return this.page.getByRole('button', { name: /^add tool$/i });
  }

  /** "Bind Tools to Selected Agent" button. */
  get bindToolsButton(): Locator {
    return this.page.getByRole('button', { name: /bind tools/i });
  }

  /** Empty-state message in the ToolsPanel. */
  get noToolsDeclaredText(): Locator {
    return this.page.getByText(/no tools declared yet/i);
  }

  // ---------------------------------------------------------------------------
  // AddToolDialog locators (Radix Dialog, role="dialog")
  // ---------------------------------------------------------------------------

  /** The AddToolDialog container. */
  get addToolDialog(): Locator {
    return this.page.getByRole('dialog');
  }

  /** Tool name input inside the AddToolDialog (id="tool-name-input"). */
  get toolNameInput(): Locator {
    return this.addToolDialog.getByLabel(/^tool name/i);
  }

  /** "Add Tool" submit button inside the AddToolDialog footer. */
  get addToolSubmitButton(): Locator {
    return this.addToolDialog.getByRole('button', { name: /^add tool$/i });
  }

  // ---------------------------------------------------------------------------
  // High-level actions
  // ---------------------------------------------------------------------------

  /** Fill the agent name input. */
  async setName(name: string): Promise<void> {
    await this.nameInput.fill(name);
  }

  /** Fill the agent description input. */
  async setDescription(description: string): Promise<void> {
    await this.descriptionInput.fill(description);
  }

  /** Click the Save button. */
  async save(): Promise<void> {
    await this.saveButton.click();
  }

  /**
   * Click "Add Root" to initialise a default sequence root in an empty workflow.
   * Only available when the BlockEditor is in its empty (ast === null) state.
   */
  async addRootBlock(): Promise<void> {
    await this.addRootButton.click();
  }

  /** Add a child block to the root workflow sequence using the visible node label. */
  async addRootChildBlock(nodeLabel: string): Promise<void> {
    await this.page.getByRole('button', { name: /^add block$/i }).first().click();
    await this.page.getByRole('menuitem', { name: new RegExp(`^${nodeLabel}$`, 'i') }).click();
  }

  /**
   * Declare a new tool via the AddToolDialog.
   *
   * Steps:
   *  1. Click "Add Tool" to open the dialog.
   *  2. Click the kind card whose visible label matches `kindLabel`.
   *  3. Clear the auto-filled name and type `toolName`.
   *  4. Click "Add Tool" to submit.
   *
   * @param kindLabel  Visible label of the tool kind (e.g., "Web Search").
   * @param toolName   Unique name for this tool instance.
   */
  async declareTool(kindLabel: string, toolName: string): Promise<void> {
    await this.addToolButton.click();
    await this.addToolDialog.waitFor({ state: 'visible' });

    // Pick the kind card
    await this.addToolDialog
      .getByRole('button', { name: new RegExp(kindLabel, 'i') })
      .first()
      .click();

    // Clear auto-filled name and type new name
    await this.toolNameInput.clear();
    await this.toolNameInput.fill(toolName);

    await this.addToolSubmitButton.click();
    await this.addToolDialog.waitFor({ state: 'hidden' });
  }

  // ---------------------------------------------------------------------------
  // Assertions
  // ---------------------------------------------------------------------------

  /**
   * Wait for the URL to transition to /designer/{id} (non-new) after a successful save.
   */
  async assertSavedSuccessfully(): Promise<void> {
    await this.page.waitForURL(/\/designer\/(?!new)[^/]+/);
  }

  /**
   * Assert the page is on the designer list (/designer).
   */
  async assertOnList(): Promise<void> {
    await expect(this.page).toHaveURL(/\/designer\/?$/);
  }

  /**
   * Assert the "Saved" status indicator is visible (isDirty === false).
   */
  async assertSavedStatus(): Promise<void> {
    await expect(this.savedIndicator).toBeVisible();
  }

  /**
   * Assert a tool row with the given name is visible in the ToolsPanel.
   * @param toolName The tool name to look for.
   */
  async assertToolDeclared(toolName: string): Promise<void> {
    await expect(this.page.getByText(toolName)).toBeVisible();
  }
}
